import torch
import torch.nn as nn
from mmdet.models import DETECTORS

from .uniad_e2e import UniADTRT


class AgentQueryFusionTRT(nn.Module):
    """TensorRT-port parameter layout for UniV2X agent query fusion."""

    def __init__(
        self, pc_range, embed_dims=256, normalization_epsilon=0.0009765625
    ):
        super().__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.normalization_epsilon = normalization_epsilon
        self.get_pos_embedding = nn.Linear(3, embed_dims)
        self.cross_agent_align = nn.Linear(embed_dims + 9, embed_dims)
        self.cross_agent_align_pos = nn.Linear(embed_dims + 9, embed_dims)
        self.cross_agent_fusion = nn.Linear(embed_dims, embed_dims)

    @staticmethod
    def _denormalize(ref_pts, pc_range):
        scale = ref_pts.new_tensor([
            pc_range[3] - pc_range[0],
            pc_range[4] - pc_range[1],
            pc_range[5] - pc_range[2],
        ])
        offset = ref_pts.new_tensor(pc_range[:3])
        return ref_pts.sigmoid() * scale + offset

    def _normalize(self, locs, pc_range):
        scale = locs.new_tensor([
            pc_range[3] - pc_range[0],
            pc_range[4] - pc_range[1],
            pc_range[5] - pc_range[2],
        ])
        offset = locs.new_tensor(pc_range[:3])
        eps = self.normalization_epsilon
        value = ((locs - offset) / scale).clamp(eps, 1.0 - eps)
        return torch.log(value / (1.0 - value))

    def forward_trt(
        self,
        inf_tracks,
        veh_tracks,
        match_vehicle_index,
        other2ego_rt,
        other_pc_range,
    ):
        inf_locs = self._denormalize(inf_tracks[1], other_pc_range)
        homogeneous = torch.cat(
            [inf_locs, torch.ones_like(inf_locs[:, :1])], dim=-1
        )
        inf_locs = torch.matmul(
            other2ego_rt[0], homogeneous.unsqueeze(-1)
        ).squeeze(-1)[:, :3]
        inf_ref_pts = self._normalize(inf_locs, self.pc_range)
        veh_ref_pts = self._normalize(
            self._denormalize(veh_tracks[1], self.pc_range), self.pc_range
        )

        rotation = other2ego_rt[0, :3, :3].reshape(1, 9)
        rotation = rotation.expand(inf_tracks[0].shape[0], -1)
        inf_query_pos = self.cross_agent_align_pos(
            torch.cat([inf_tracks[0][:, :self.embed_dims], rotation], dim=-1)
        )
        inf_query_feat = self.cross_agent_align(
            torch.cat([inf_tracks[0][:, self.embed_dims:], rotation], dim=-1)
        )
        inf_query = torch.cat([inf_query_pos, inf_query_feat], dim=-1)

        matched = match_vehicle_index >= 0
        safe_match = match_vehicle_index.clamp(min=0).long()
        vehicle_range = torch.arange(
            veh_tracks[0].shape[0], device=safe_match.device
        )
        assignment = (
            safe_match[:, None] == vehicle_range[None, :]
        ) & matched[:, None]
        delta = torch.matmul(
            assignment.to(inf_query_feat.dtype).transpose(0, 1),
            self.cross_agent_fusion(inf_query_feat),
        )
        veh_query = torch.cat(
            [
                veh_tracks[0][:, :self.embed_dims],
                veh_tracks[0][:, self.embed_dims:] + delta,
            ],
            dim=-1,
        )

        unmatched = ~matched
        transformed_inf = list(inf_tracks)
        transformed_inf[0] = inf_query
        transformed_inf[1] = inf_ref_pts
        transformed_inf[3] = torch.full_like(inf_tracks[3], -1)
        fused = []
        for index, (veh_value, inf_value) in enumerate(
            zip(veh_tracks, transformed_inf)
        ):
            if index == 0:
                veh_value = veh_query
            elif index == 1:
                veh_value = veh_ref_pts
            fused.append(torch.cat([veh_value, inf_value[unmatched]], dim=0))
        return fused, inf_query[unmatched], inf_ref_pts[unmatched]


class LaneQueryFusionTRT(nn.Module):
    """TensorRT-port parameter layout for UniV2X lane query fusion."""

    def __init__(self, pc_range, embed_dims=256):
        super().__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.get_pos_embedding = nn.Linear(3, embed_dims)
        self.cross_agent_align = nn.Linear(embed_dims + 9, embed_dims)
        self.cross_agent_align_pos = nn.Linear(embed_dims + 9, embed_dims)
        self.cross_agent_fusion = nn.Linear(embed_dims, embed_dims)

    @staticmethod
    def _transform_points(points, source_range, target_range, transform,
                          inverse_sigmoid_mode):
        if inverse_sigmoid_mode:
            points = points.sigmoid()
        source_scale = points.new_tensor([
            source_range[3] - source_range[0],
            source_range[4] - source_range[1],
            source_range[5] - source_range[2],
        ])
        source_offset = points.new_tensor(source_range[:3])
        locs = points * source_scale + source_offset
        homogeneous = torch.cat(
            [locs, torch.ones_like(locs[..., :1])], dim=-1
        )
        locs = torch.matmul(
            transform[0], homogeneous.unsqueeze(-1)
        ).squeeze(-1)[..., :3]
        target_scale = points.new_tensor([
            target_range[3] - target_range[0],
            target_range[4] - target_range[1],
            target_range[5] - target_range[2],
        ])
        target_offset = points.new_tensor(target_range[:3])
        normalized = ((locs - target_offset) / target_scale).clamp(
            1e-5, 1.0 - 1e-5
        )
        if inverse_sigmoid_mode:
            normalized = torch.log(normalized / (1.0 - normalized))
        return normalized

    def forward_trt(
        self,
        other_outputs_classes,
        other_outputs_coords,
        other_query,
        other_query_pos,
        other_reference,
        veh_outputs_classes,
        veh_outputs_coords,
        veh_query,
        veh_query_pos,
        veh_reference,
        other2ego_rt,
        other_pc_range,
    ):
        indexes = torch.where(
            other_outputs_classes[-1].sigmoid().reshape(-1) > 0.05
        )[0]
        bbox_index = torch.div(indexes, 3, rounding_mode="floor")
        other_outputs_classes = other_outputs_classes[:, :, bbox_index]
        other_outputs_coords = other_outputs_coords[:, :, bbox_index]
        other_query = other_query[:, bbox_index]
        other_query_pos = other_query_pos[:, bbox_index]
        other_reference = other_reference[:, bbox_index]

        reference_xyz = torch.cat(
            [
                other_reference[..., :2],
                torch.zeros_like(other_reference[..., :1]),
            ],
            dim=-1,
        )
        transformed_reference = self._transform_points(
            reference_xyz,
            other_pc_range,
            self.pc_range,
            other2ego_rt,
            True,
        )
        other_reference = torch.cat(
            [transformed_reference[..., :2], other_reference[..., 2:]], dim=-1
        )

        coords_xyz = torch.cat(
            [
                other_outputs_coords[..., :2],
                torch.zeros_like(other_outputs_coords[..., :1]),
            ],
            dim=-1,
        )
        transformed_coords = self._transform_points(
            coords_xyz,
            other_pc_range,
            self.pc_range,
            other2ego_rt,
            False,
        )
        other_outputs_coords = torch.cat(
            [transformed_coords[..., :2], other_outputs_coords[..., 2:]],
            dim=-1,
        )

        rotation = other2ego_rt[0, :3, :3].reshape(1, 1, 9)
        rotation = rotation.expand(other_query.shape[0], other_query.shape[1], -1)
        other_query = self.cross_agent_align(
            torch.cat([other_query, rotation], dim=-1)
        )
        other_query_pos = self.cross_agent_align_pos(
            torch.cat([other_query_pos, rotation], dim=-1)
        )
        return (
            torch.cat([veh_outputs_classes, other_outputs_classes], dim=2),
            torch.cat([veh_outputs_coords, other_outputs_coords], dim=2),
            torch.cat([veh_query, other_query], dim=1),
            torch.cat([veh_query_pos, other_query_pos], dim=1),
            torch.cat([veh_reference, other_reference], dim=1),
            other_query,
            other_query_pos,
            other_reference,
        )


@DETECTORS.register_module()
class UniV2XTRT(UniADTRT):
    """Exportable UniAD graph plus the learned UniV2X cooperative modules."""

    def __init__(
        self,
        univ2x_is_cooperation=False,
        univ2x_is_ego_agent=False,
        univ2x_inf_pc_range=None,
        univ2x_seg_bev_aug=False,
        univ2x_occ_pred_prob=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.is_cooperation = univ2x_is_cooperation
        self.is_ego_agent = univ2x_is_ego_agent
        self.inf_pc_range = univ2x_inf_pc_range

        self.bev_embed_linear = nn.Linear(self.embed_dims, self.embed_dims)
        self.bev_pos_linear = nn.Linear(self.embed_dims, self.embed_dims)

        if self.is_cooperation:
            self.cross_agent_query_interaction = AgentQueryFusionTRT(
                pc_range=self.pc_range, embed_dims=self.embed_dims
            )

        if self.with_seg_head and univ2x_seg_bev_aug:
            self.seg_head.bev_embed_linear = nn.Linear(
                self.embed_dims, self.embed_dims
            )
            self.seg_head.bev_pos_linear = nn.Linear(
                self.embed_dims, self.embed_dims
            )
        if self.with_seg_head and self.is_cooperation:
            self.seg_head.cross_lane_fusion = LaneQueryFusionTRT(
                pc_range=self.pc_range, embed_dims=self.embed_dims
            )

        if self.with_occ_head and univ2x_occ_pred_prob:
            self.occ_head.pred_prob = nn.ModuleList(
                [
                    nn.Conv2d(self.embed_dims, 1, kernel_size=1)
                    for _ in range(self.occ_head.n_future_blocks)
                ]
            )
        if self.with_occ_head:
            self.occ_head.univ2x_pc_range = self.pc_range
            self.occ_head.univ2x_inf_pc_range = self.inf_pc_range

    @staticmethod
    def _scatter_bev(bev, query, reference, projection, radius):
        if query.shape[0] == 0:
            return bev
        height = width = 200
        xy = (reference[:, :2].sigmoid() * reference.new_tensor(
            [width, height]
        )).long()
        axis = torch.arange(-radius, radius, device=xy.device)
        offsets = torch.stack(
            [
                axis[:, None].expand(-1, axis.shape[0]).reshape(-1),
                axis[None, :].expand(axis.shape[0], -1).reshape(-1),
            ],
            dim=-1,
        )
        locations = xy[:, None, [1, 0]] + offsets[None]
        valid = (
            (locations[..., 0] >= 0)
            & (locations[..., 0] < height - 1)
            & (locations[..., 1] >= 0)
            & (locations[..., 1] < width - 1)
        )
        flat_locations = (
            locations[..., 0] * width + locations[..., 1]
        )[valid]
        updates = projection(query)[:, None].expand(-1, offsets.shape[0], -1)[valid]
        flattened = bev.reshape(-1, bev.shape[-1])
        return flattened.index_add(0, flat_locations, updates).reshape_as(bev)

    def fuse_agent_tracks_trt(
        self, inf_tracks, veh_tracks, match_vehicle_index, other2ego_rt
    ):
        fused, added_query, added_reference = (
            self.cross_agent_query_interaction.forward_trt(
                inf_tracks,
                veh_tracks,
                match_vehicle_index,
                other2ego_rt,
                self.inf_pc_range,
            )
        )
        return fused, added_query, added_reference

    def augment_track_bev_trt(
        self, bev_embed, bev_pos, added_query, added_reference
    ):
        bev_embed = self._scatter_bev(
            bev_embed,
            added_query[:, self.embed_dims:],
            added_reference,
            self.bev_embed_linear,
            1,
        )
        pos_flat = bev_pos.permute(0, 2, 3, 1).reshape(-1, self.embed_dims)
        pos_flat = self._scatter_bev(
            pos_flat,
            added_query[:, :self.embed_dims],
            added_reference,
            self.bev_pos_linear,
            1,
        )
        bev_pos = pos_flat.reshape(1, 200, 200, self.embed_dims).permute(0, 3, 1, 2)
        return bev_embed, bev_pos

    def select_fixed_motion_queries_trt(self, track_instances, capacity=64):
        active = (
            (track_instances[3] >= 0)
            & (track_instances[7] >= self.track_base.filter_score_thresh)
        )
        active_index = self.index_bool2long_trt(active)
        active_instances = [value[active_index] for value in track_instances]
        (
            _boxes,
            gravity_center,
            yaw,
            _scores,
            _labels,
            _track_scores,
            bbox_index,
            _obj_idxes,
            valid_box_mask,
            track_scores,
            labels,
        ) = self._track_instances2results_trt(
            active_instances, with_mask=True
        )
        selected = bbox_index[self.index_bool2long_trt(valid_box_mask)]
        query_embeddings = active_instances[2][selected]

        # Preserve the original ordering: every active object participates in
        # the motion transformer, then the motion head filters vehicle outputs
        # before occupancy. Padding keeps the TensorRT motion extent static.
        rank_scores = track_scores
        rank_scores = torch.cat(
            [rank_scores, rank_scores.new_zeros((capacity,))], dim=0
        )
        query_embeddings = torch.cat(
            [
                query_embeddings,
                query_embeddings.new_zeros((capacity, query_embeddings.shape[-1])),
            ],
            dim=0,
        )
        track_scores = torch.cat(
            [
                track_scores,
                track_scores.new_zeros((capacity,)),
            ],
            dim=0,
        )
        # Class 0 is a vehicle. Mark padding with an invalid class so the
        # motion/occupancy vehicle filter cannot treat empty slots as cars.
        labels = torch.cat(
            [labels, labels.new_full((capacity,), -1)], dim=0
        )
        gravity_center = torch.cat(
            [
                gravity_center,
                gravity_center.new_zeros((capacity, gravity_center.shape[-1])),
            ],
            dim=0,
        )
        yaw = torch.cat([yaw, yaw.new_zeros((capacity,))], dim=0)
        fixed_index = torch.topk(rank_scores, capacity, sorted=True).indices
        return (
            query_embeddings[fixed_index],
            track_scores[fixed_index],
            labels[fixed_index],
            gravity_center[fixed_index],
            yaw[fixed_index],
        )

    @staticmethod
    def pad_fixed_motion_queries_trt(
        query_embeddings,
        track_scores,
        labels,
        gravity_center,
        yaw,
        capacity=64,
    ):
        """Pad current-frame motion inputs without re-reading memory-bank state."""
        query_embeddings = torch.cat(
            [
                query_embeddings,
                query_embeddings.new_zeros(
                    (capacity, query_embeddings.shape[-1])
                ),
            ],
            dim=0,
        )[:capacity]
        track_scores = torch.cat(
            [track_scores, track_scores.new_zeros((capacity,))], dim=0
        )[:capacity]
        labels = torch.cat(
            [labels, labels.new_full((capacity,), -1)], dim=0
        )[:capacity]
        gravity_center = torch.cat(
            [
                gravity_center,
                gravity_center.new_zeros(
                    (capacity, gravity_center.shape[-1])
                ),
            ],
            dim=0,
        )[:capacity]
        yaw = torch.cat([yaw, yaw.new_zeros((capacity,))], dim=0)[:capacity]
        return query_embeddings, track_scores, labels, gravity_center, yaw
