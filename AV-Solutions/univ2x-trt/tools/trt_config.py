from copy import deepcopy
import os


TYPE_MAP = {
    "UniV2X": "UniV2XTRT",
    "BEVFormerTrackHead": "BEVFormerTrackHeadTRTP",
    "PerceptionTransformer": "PerceptionTransformerUniADTRTP",
    "BEVFormerEncoder": "BEVFormerEncoderTRTP",
    "BEVFormerLayer": "BEVFormerLayerTRTP",
    "TemporalSelfAttention": "TemporalSelfAttentionTRTP",
    "SpatialCrossAttention": "SpatialCrossAttentionTRTP",
    "MSDeformableAttention3D": "MSDeformableAttention3DTRTP",
    "DetectionTransformerDecoder": "DetectionTransformerDecoderTRTP",
    "CustomMSDeformableAttention": "CustomMSDeformableAttentionTRTP",
    "PansegformerHead": "PansegformerHeadTRTP",
    "MultiScaleDeformableAttention": "MultiScaleDeformableAttentionTRTP",
    "OccHead": "OccHeadTRTP",
    "MotionHead": "MotionHeadTRTP",
    "MotionTransformerDecoder": "MotionTransformerDecoderTRTP",
    "MotionTransformerAttentionLayer": "MotionTransformerAttentionLayerTRTP",
    "MotionDeformableAttention": "MotionDeformableAttentionTRTP",
    "PlanningHeadSingleMode": "PlanningHeadSingleModeTRTP",
}


def _map_types(value):
    if isinstance(value, dict):
        mapped = {key: _map_types(item) for key, item in value.items()}
        if "type" in mapped:
            mapped["type"] = TYPE_MAP.get(mapped["type"], mapped["type"])
        return value.__class__(mapped)
    if isinstance(value, list):
        return [_map_types(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_types(item) for item in value)
    return value


def make_trt_agent_config(source, cooperative=False):
    """Map a UniV2X agent config to the NVIDIA UniAD TensorRT operators.

    Cooperative-only modules are attached by the UniV2X TRT wrapper after the
    inherited exportable UniAD graph has been validated.
    """
    source = deepcopy(source)
    is_ego_agent = bool(source.get("is_ego_agent", False))
    inf_pc_range = deepcopy(source.get("inf_pc_range"))
    seg_bev_aug = bool(source.get("seg_head", {}).get("is_bev_aug", False))
    occ_pred_prob = not bool(source.get("occ_head", {}).get("is_old_mode", False))
    cfg = _map_types(deepcopy(dict(source)))

    pc_range = cfg["pc_range"]
    post_center_range = cfg.pop(
        "post_center_range",
        [pc_range[0] - 10.0, pc_range[1] - 10.0, -10.0,
         pc_range[3] + 10.0, pc_range[4] + 10.0, 10.0],
    )
    cfg["bbox_coder"] = {
        "type": "DETRTrack3DCoder",
        "post_center_range": post_center_range,
        "pc_range": pc_range,
        "max_num": 300,
        "num_classes": cfg.get("num_classes", 10),
        "score_threshold": 0.0,
        "with_nms": False,
        "iou_thres": 0.3,
    }

    for key in (
        "is_cooperation",
        "is_ego_agent",
        "inf_pc_range",
        "return_track_query",
        "save_track_query",
        "save_track_query_file_root",
        "load_from",
    ):
        cfg.pop(key, None)

    seg_head = cfg.get("seg_head")
    if seg_head:
        for key in ("inf_pc_range", "is_cooperation", "is_bev_aug"):
            seg_head.pop(key, None)

    occ_head = cfg.get("occ_head")
    if occ_head:
        for key in (
            "bev_h",
            "bev_w",
            "pc_range",
            "inf_pc_range",
            "is_cooperation",
            "is_ego_agent",
            "is_old_mode",
            "return_occ_data",
        ):
            occ_head.pop(key, None)

    motion_head = cfg.get("motion_head")
    if motion_head and motion_head.get("anchor_info_path"):
        anchor_path = motion_head["anchor_info_path"]
        if not os.path.isabs(anchor_path):
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            motion_head["anchor_info_path"] = os.path.join(
                project_root, anchor_path.lstrip("./")
            )

    cfg["univ2x_cooperative_graph"] = bool(cooperative)
    cfg["univ2x_is_cooperation"] = bool(cooperative)
    cfg["univ2x_is_ego_agent"] = is_ego_agent
    cfg["univ2x_inf_pc_range"] = inf_pc_range
    cfg["univ2x_seg_bev_aug"] = seg_bev_aug
    cfg["univ2x_occ_pred_prob"] = occ_pred_prob
    return cfg
