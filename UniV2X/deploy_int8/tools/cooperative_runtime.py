from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from trt_runtime import (
    COOP_INPUT_NAMES,
    INPUT_NAMES,
    TRACK_OUTPUT_NAMES,
    empty_track_state,
)


INFRASTRUCTURE_PC_RANGE = [0.0, -51.2, -5.0, 102.4, 51.2, 3.0]
EGO_PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def first_tensor(value):
    while isinstance(value, (list, tuple)):
        value = value[0]
    return value


def metadata(agent_data):
    return agent_data["img_metas"][0].data[0][0]


def agent_tensor(agent_data, key, device, dtype=None):
    value = first_tensor(agent_data[key])
    if hasattr(value, "data") and value.__class__.__name__ == "DataContainer":
        value = first_tensor(value.data)
    value = value.to(device=device)
    return value.to(dtype=dtype) if dtype is not None else value


def agent_image(agent_data, device):
    value = first_tensor(agent_data["img"])
    if hasattr(value, "data") and value.__class__.__name__ == "DataContainer":
        value = first_tensor(value.data)
    return value.to(device=device, dtype=torch.float32)


def denormalize(reference, pc_range):
    scale = reference.new_tensor([
        pc_range[3] - pc_range[0],
        pc_range[4] - pc_range[1],
        pc_range[5] - pc_range[2],
    ])
    offset = reference.new_tensor(pc_range[:3])
    return reference.sigmoid() * scale + offset


@dataclass
class AgentState:
    device: torch.device
    fixed_track_count: int = 0

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.tracks = empty_track_state(self.device)
        self.prev_timestamp = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.timestamp_origin = None
        self.prev_l2g_r_mat = torch.zeros(1, 3, 3, dtype=torch.float32, device=self.device)
        self.prev_l2g_t = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.prev_bev = torch.zeros(40000, 1, 256, dtype=torch.float32, device=self.device)
        self.max_obj_id = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.scene_token = None
        self.prev_position = None
        self.prev_angle = None

    def build_inputs(self, agent_data, include_ground_truth=True):
        meta = metadata(agent_data)
        scene_token = str(meta["scene_token"])
        new_scene = scene_token != self.scene_token
        if new_scene:
            self.reset()

        image = agent_image(agent_data, self.device)
        lidar2img = np.asarray(meta["lidar2img"], dtype=np.float32)[None]
        if lidar2img.shape[1] == 1:
            lidar2img = np.repeat(lidar2img, 6, axis=1)
        can_bus_now = np.asarray(meta["can_bus"], dtype=np.float32).copy()
        can_bus = can_bus_now.copy()
        if new_scene or self.prev_position is None:
            can_bus[:3] = 0.0
            can_bus[-1] = 0.0
        else:
            can_bus[:3] -= self.prev_position
            can_bus[-1] -= self.prev_angle
        self.prev_position = can_bus_now[:3].copy()
        self.prev_angle = float(can_bus_now[-1])
        self.scene_token = scene_token

        timestamp = agent_tensor(agent_data, "timestamp", self.device)
        if self.timestamp_origin is None:
            self.timestamp_origin = timestamp.clone()
        timestamp = (timestamp - self.timestamp_origin).to(dtype=torch.float32)

        scene_bytes = np.zeros(32, dtype=np.uint8)
        encoded = scene_token.encode("utf-8")[:32]
        scene_bytes[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        image_shape = torch.tensor(
            image.shape[-2:], dtype=torch.float32, device=self.device
        )
        if include_ground_truth:
            ground_truth = [
                agent_tensor(agent_data, "gt_lane_labels", self.device, torch.int64),
                agent_tensor(agent_data, "gt_lane_masks", self.device, torch.uint8),
                agent_tensor(agent_data, "gt_segmentation", self.device, torch.int64),
            ]
        else:
            ground_truth = [
                torch.empty(0, dtype=torch.int64, device=self.device),
                torch.empty(0, dtype=torch.uint8, device=self.device),
                torch.empty(0, dtype=torch.int64, device=self.device),
            ]

        input_tracks = self.tracks
        if self.fixed_track_count:
            current = int(input_tracks[0].shape[0])
            if current > self.fixed_track_count:
                raise RuntimeError(
                    "Track count %d exceeds fixed input capacity %d"
                    % (current, self.fixed_track_count)
                )
            if current < self.fixed_track_count:
                input_tracks = [
                    torch.cat([
                        value,
                        torch.full(
                            (self.fixed_track_count - current,) + value.shape[1:],
                            -10000,
                            dtype=value.dtype,
                            device=value.device,
                        ),
                    ], dim=0)
                    for value in input_tracks
                ]

        values = [
            *input_tracks,
            self.prev_timestamp,
            self.prev_l2g_r_mat,
            self.prev_l2g_t,
            self.prev_bev,
            *ground_truth,
            torch.from_numpy(scene_bytes).to(self.device),
            timestamp,
            agent_tensor(agent_data, "l2g_r_mat", self.device, torch.float32),
            agent_tensor(agent_data, "l2g_t", self.device, torch.float32),
            image,
            torch.from_numpy(can_bus).to(self.device),
            torch.from_numpy(lidar2img).to(self.device),
            image_shape,
            agent_tensor(agent_data, "command", self.device, torch.int64),
            torch.tensor([0 if new_scene else 1], dtype=torch.int32, device=self.device),
            self.max_obj_id,
        ]
        return dict(zip(INPUT_NAMES, values)), new_scene

    def update(self, outputs, timestamp_fallback=None):
        self.tracks = [outputs[name] for name in TRACK_OUTPUT_NAMES]
        timestamp = outputs["prev_timestamp_out"]
        if not torch.isfinite(timestamp).all():
            if timestamp_fallback is None or not torch.isfinite(timestamp_fallback).all():
                raise RuntimeError("Non-finite prev_timestamp_out without finite fallback")
            timestamp = timestamp_fallback
        self.prev_timestamp = timestamp
        self.prev_l2g_t = outputs["prev_l2g_t_out"]
        self.prev_l2g_r_mat = outputs["prev_l2g_r_mat_out"]
        self.prev_bev = outputs["bev_embed"]
        self.max_obj_id = outputs["max_obj_id_out"]


def prepare_cooperative_inputs(
    infrastructure_outputs, ego_state, veh2inf_rt, fixed_coop_count=0
):
    tracks = [infrastructure_outputs[name] for name in TRACK_OUTPUT_NAMES]
    active = tracks[3] >= 0
    tracks = [value[active] for value in tracks]

    ego2other = veh2inf_rt.to(dtype=torch.float32)
    other2ego_np = np.linalg.inv(ego2other[0].detach().cpu().numpy().T).astype(np.float32)
    other2ego = tracks[0].new_tensor(other2ego_np)[None]
    infrastructure_locations = denormalize(tracks[1], INFRASTRUCTURE_PC_RANGE)
    homogeneous = torch.cat(
        [infrastructure_locations, torch.ones_like(infrastructure_locations[:, :1])],
        dim=-1,
    )
    ego_locations = torch.matmul(
        other2ego[0], homogeneous.unsqueeze(-1)
    ).squeeze(-1)[:, :3]

    inside_ego = (
        (ego_locations[:, 0] >= -2.04)
        & (ego_locations[:, 0] <= 2.04)
        & (ego_locations[:, 1] >= -0.92)
        & (ego_locations[:, 1] <= 0.92)
    )
    keep = torch.ones_like(inside_ego)
    candidates = torch.where(inside_ego)[0]
    if candidates.numel() > 0:
        keep[candidates[0]] = False
    tracks = [value[keep] for value in tracks]
    ego_locations = ego_locations[keep]

    if tracks[0].shape[0] == 0:
        raise RuntimeError("No active infrastructure query remains after ego removal")

    vehicle_locations = denormalize(ego_state.tracks[1], EGO_PC_RANGE)
    vehicle_scores = ego_state.tracks[7]
    vehicle_dims = ego_state.tracks[9][:, [2, 3, 5]]
    vehicle_candidates = torch.where(vehicle_scores >= 0.05)[0]
    costs = np.full(
        (vehicle_locations.shape[0], ego_locations.shape[0]),
        1.0e6,
        dtype=np.float64,
    )
    if vehicle_candidates.numel() > 0:
        vehicle_np = vehicle_locations[vehicle_candidates].detach().cpu().numpy()
        dimensions_np = vehicle_dims[vehicle_candidates].detach().cpu().numpy()
        infrastructure_np = ego_locations.detach().cpu().numpy()
        for row, vehicle_index in enumerate(vehicle_candidates.detach().cpu().numpy()):
            difference = vehicle_np[row][None] - infrastructure_np
            distance = np.linalg.norm(difference, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized_difference = (
                    np.abs(difference) / dimensions_np[row][None]
                )
            valid = np.all(normalized_difference <= 1.0, axis=1)
            costs[int(vehicle_index), valid] = distance[valid]

    vehicle_indices, infrastructure_indices = linear_sum_assignment(costs)
    match = torch.full(
        (tracks[0].shape[0],), -1, dtype=torch.int64, device=tracks[0].device
    )
    accepted = costs[vehicle_indices, infrastructure_indices] < 1.0e5
    if accepted.any():
        match[torch.as_tensor(
            infrastructure_indices[accepted], device=match.device
        )] = torch.as_tensor(vehicle_indices[accepted], device=match.device)

    cooperative_track_count = int(tracks[0].shape[0])
    if fixed_coop_count:
        if cooperative_track_count > fixed_coop_count:
            raise RuntimeError(
                "Cooperative track count %d exceeds fixed input capacity %d"
                % (cooperative_track_count, fixed_coop_count)
            )
        padding = fixed_coop_count - cooperative_track_count
        if padding:
            tracks = [
                torch.cat([
                    value,
                    torch.zeros(
                        (padding,) + value.shape[1:],
                        dtype=value.dtype,
                        device=value.device,
                    ),
                ], dim=0)
                for value in tracks
            ]
            match = torch.cat([
                match,
                torch.full(
                    (padding,), 2147483647,
                    dtype=match.dtype,
                    device=match.device,
                ),
            ])

    values = {
        **{f"coop_track_intances{index}": value for index, value in enumerate(tracks)},
        "coop_match_vehicle_index": match,
        "coop_other2ego_rt": other2ego,
        "coop_lane_outputs_classes": infrastructure_outputs["lane_outputs_classes"],
        "coop_lane_outputs_coords": infrastructure_outputs["lane_outputs_coords"],
        "coop_lane_query": infrastructure_outputs["lane_query"],
        "coop_lane_query_pos": infrastructure_outputs["lane_query_pos"],
        "coop_lane_reference": infrastructure_outputs["lane_reference"],
        "coop_occ_pred_scores": infrastructure_outputs["occ_pred_scores"],
        "coop_ego2other_rt": ego2other,
    }
    if list(values) != COOP_INPUT_NAMES:
        raise AssertionError((list(values), COOP_INPUT_NAMES))
    return values, {
        "active_before_ego_removal": int(active.sum().item()),
        "ego_instances_removed": int((~keep).sum().item()),
        "cooperative_track_count": cooperative_track_count,
        "cooperative_input_count": int(tracks[0].shape[0]),
        "matched_track_count": int(accepted.sum()),
    }
