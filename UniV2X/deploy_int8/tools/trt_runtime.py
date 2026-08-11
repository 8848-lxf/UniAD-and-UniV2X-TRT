import gc

import numpy as np
import torch

from trt_config import make_trt_agent_config


INPUT_NAMES = [
    *[f"prev_track_intances{i}" for i in range(14)],
    "prev_timestamp",
    "prev_l2g_r_mat",
    "prev_l2g_t",
    "prev_bev",
    "gt_lane_labels",
    "gt_lane_masks",
    "gt_segmentation",
    "img_metas_scene_token",
    "timestamp",
    "l2g_r_mat",
    "l2g_t",
    "img",
    "img_metas_can_bus",
    "img_metas_lidar2img",
    "image_shape",
    "command",
    "use_prev_bev",
    "max_obj_id",
]

COOP_TRACK_INPUT_NAMES = [
    *[f"coop_track_intances{index}" for index in range(14)],
    "coop_match_vehicle_index",
    "coop_other2ego_rt",
]
COOP_TASK_INPUT_NAMES = [
    "coop_lane_outputs_classes",
    "coop_lane_outputs_coords",
    "coop_lane_query",
    "coop_lane_query_pos",
    "coop_lane_reference",
    "coop_occ_pred_scores",
    "coop_ego2other_rt",
]
COOP_INPUT_NAMES = COOP_TRACK_INPUT_NAMES + COOP_TASK_INPUT_NAMES
EGO_INPUT_NAMES = INPUT_NAMES + COOP_INPUT_NAMES

TRACK_OUTPUT_NAMES = [
    f"prev_track_intances{index}_out" for index in range(14)
]

TASK_OUTPUT_NAMES = [
    "drivable_pred",
    "lane_pred",
    "map_raw_masks",
    "map_raw_scores",
    "map_raw_labels",
    "drivable_intersection",
    "drivable_union",
    "lanes_intersection",
    "lanes_union",
    "divider_intersection",
    "divider_union",
    "crossing_intersection",
    "crossing_union",
    "contour_intersection",
    "contour_union",
    "lane_outputs_classes",
    "lane_outputs_coords",
    "lane_query",
    "lane_query_pos",
    "lane_reference",
    "motion_traj_scores",
    "motion_trajs",
    "motion_valid_masks",
    "occ_pred_scores",
    "occ_pred_sigmoid",
    "seg_out",
]

OUTPUT_NAMES = [
    *TRACK_OUTPUT_NAMES,
    "prev_timestamp_out",
    "prev_l2g_t_out",
    "prev_l2g_r_mat_out",
    "bev_embed",
    "bboxes_dict_bboxes",
    "scores",
    "labels",
    "bbox_index",
    "obj_idxes",
    "det_bboxes",
    "det_scores",
    "det_labels",
    "max_obj_id_out",
    *TASK_OUTPUT_NAMES,
    "outs_planning",
]

INFRASTRUCTURE_OUTPUT_NAMES = OUTPUT_NAMES[:-1]

RAW_MAP_OUTPUT_NAMES = ("map_raw_masks", "map_raw_scores", "map_raw_labels")


def apply_dynamic_map_postprocess(outputs):
    present = [name in outputs for name in RAW_MAP_OUTPUT_NAMES]
    if not any(present):
        return False
    if not all(present):
        raise KeyError("Incomplete raw map outputs: %s" % dict(zip(
            RAW_MAP_OUTPUT_NAMES, present
        )))

    raw_masks = outputs["map_raw_masks"]
    raw_scores = outputs["map_raw_scores"]
    raw_labels = outputs["map_raw_labels"]
    if raw_masks.ndim != 3 or tuple(raw_masks.shape[1:]) != (200, 200):
        raise ValueError("Unexpected map_raw_masks shape: %s" % (raw_masks.shape,))
    query_count = raw_masks.shape[0]
    if tuple(raw_scores.shape) != (query_count,):
        raise ValueError("Unexpected map_raw_scores shape: %s" % (raw_scores.shape,))
    if tuple(raw_labels.shape) != (query_count,):
        raise ValueError("Unexpected map_raw_labels shape: %s" % (raw_labels.shape,))

    masks = raw_masks.detach().float().cpu().numpy()
    scores = raw_scores.detach().float().cpu().numpy()
    labels = raw_labels.detach().long().cpu().numpy()
    panoptic = np.zeros((200, 200), dtype=np.int64)
    lane = np.zeros((3, 200, 200), dtype=np.int32)
    for index in range(query_count):
        label = int(labels[index])
        score = float(scores[index])
        if label < 3 and score < 0.1:
            continue
        if label >= 3 and score < 0.25:
            continue
        mask = masks[index] > 0.5
        mask_area = int(mask.sum())
        intersection = np.logical_and(mask, panoptic > 0)
        intersection_area = int(intersection.sum())
        overlap_limit = 0.4 if label < 3 else 0.2
        if mask_area == 0 or intersection_area / mask_area > overlap_limit:
            continue
        if intersection_area:
            mask = np.logical_and(mask, panoptic == 0)
        panoptic[mask] = label
        if label < 3:
            lane[label, mask] = 1

    traced_lane = outputs.get("lane_pred")
    if traced_lane is not None:
        outputs["lane_pred_traced"] = traced_lane
    outputs["lane_pred"] = torch.from_numpy(lane).to(
        device=raw_masks.device,
        dtype=traced_lane.dtype if traced_lane is not None else torch.int32,
    )
    return True


def build_trt_agent(config_path, checkpoint_path, agent):
    from mmcv import Config
    from third_party.uniad_mmdet3d.models.builder import build_model

    cfg = Config.fromfile(config_path)
    if agent == "ego":
        source = cfg.model_ego_agent
        prefix = "model_ego_agent."
        cooperative = True
    elif agent == "infrastructure":
        source = cfg.model_other_agent_inf
        prefix = "model_other_agent_inf."
        cooperative = False
    else:
        raise ValueError(agent)

    agent_cfg = make_trt_agent_config(source, cooperative=cooperative)
    agent_cfg.pop("univ2x_cooperative_graph")
    agent_cfg["train_cfg"] = None
    model = build_model(agent_cfg, test_cfg=cfg.get("test_cfg"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Missing checkpoint keys: {incompatible.missing_keys}")
    allowed_unexpected = set()
    if agent == "infrastructure":
        allowed_unexpected = {
            "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
            "linear_feat1.weight", "linear_feat1.bias", "linear_feat2.weight",
            "linear_feat2.bias", "norm_feat.weight", "norm_feat.bias",
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
            "self_attn.in_proj_weight", "self_attn.in_proj_bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
        }
    unexpected = set(incompatible.unexpected_keys)
    if unexpected != allowed_unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
    del checkpoint, state
    gc.collect()
    return cfg, model


def empty_track_state(device):
    query_count = 901
    embed_dims = 256
    return [
        torch.zeros(query_count, embed_dims * 2, device=device),
        torch.zeros(query_count, 3, device=device),
        torch.zeros(query_count, embed_dims, device=device),
        torch.full((query_count,), -1, dtype=torch.int32, device=device),
        torch.full((query_count,), -1, dtype=torch.int32, device=device),
        torch.zeros(query_count, dtype=torch.int32, device=device),
        torch.zeros(query_count, device=device),
        torch.zeros(query_count, device=device),
        torch.zeros(query_count, device=device),
        torch.zeros(query_count, 10, device=device),
        torch.zeros(query_count, 10, device=device),
        torch.zeros(query_count, 4, embed_dims, device=device),
        torch.ones(query_count, 4, dtype=torch.int32, device=device),
        torch.zeros(query_count, device=device),
    ]


def load_export_inputs(npz_path, device, coop_npz_path=None):
    arrays = np.load(npz_path)
    tracks = empty_track_state(device)
    bev_tokens = 200 * 200

    def tensor(name, dtype=None):
        value = torch.from_numpy(arrays[name]).to(device)
        return value.to(dtype) if dtype is not None else value

    values = [
        *tracks,
        torch.zeros(1, dtype=torch.float32, device=device),
        torch.zeros(1, 3, 3, dtype=torch.float32, device=device),
        torch.zeros(1, 3, dtype=torch.float32, device=device),
        torch.zeros(bev_tokens, 1, 256, dtype=torch.float32, device=device),
        tensor("gt_lane_labels", torch.int64),
        tensor("gt_lane_masks", torch.uint8),
        tensor("gt_segmentation", torch.int64),
        tensor("img_metas_scene_token", torch.uint8),
        tensor("timestamp", torch.float32),
        tensor("l2g_r_mat", torch.float32),
        tensor("l2g_t", torch.float32),
        tensor("img", torch.float32),
        tensor("img_metas_can_bus", torch.float32),
        tensor("img_metas_lidar2img", torch.float32),
        tensor("image_shape", torch.float32),
        tensor("command", torch.int64),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    ]
    if len(values) != len(INPUT_NAMES):
        raise AssertionError((len(values), len(INPUT_NAMES)))
    if coop_npz_path is not None:
        coop = np.load(coop_npz_path)
        values.extend(
            torch.from_numpy(coop[name]).to(device)
            for name in COOP_INPUT_NAMES
        )
    return tuple(values)
