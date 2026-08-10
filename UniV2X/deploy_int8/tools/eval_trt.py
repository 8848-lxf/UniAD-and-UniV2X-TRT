import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import sys
import time
import traceback

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mmcv
import numpy as np
import torch
from mmcv.fileio.file_client import HardDiskBackend
from mmdet3d.datasets import build_dataset

from cooperative_runtime import (
    AgentState,
    agent_tensor,
    metadata,
    prepare_cooperative_inputs,
)
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.dense_heads.occ_head_plugin import (
    IntersectionOverUnion,
    PanopticMetric,
)
from projects.mmdet3d_plugin.univ2x.dense_heads.occ_head_plugin.utils import (
    predict_instance_segmentation_and_trajectories,
)
from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head import (
    PlanningHeadSingleMode,
)
from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head_plugin import (
    PlanningMetric,
)
from runtime_common import load_config
from trt_engine import TensorRTEngine


VEHICLE_LABELS = (0, 1, 2, 3, 4, 6, 7)
MAP_RESULT_KEYS = (
    "drivable_intersection", "drivable_union",
    "lanes_intersection", "lanes_union",
    "divider_intersection", "divider_union",
    "crossing_intersection", "crossing_union",
    "contour_intersection", "contour_union",
)


def patch_disk_backend_file_objects():
    original_get = HardDiskBackend.get

    def compatible_get(backend, filepath):
        if hasattr(filepath, "read"):
            return filepath.read()
        return original_get(backend, filepath)

    HardDiskBackend.get = compatible_get


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("infrastructure_engine")
    parser.add_argument("ego_engine")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--context-cache-size", type=int, default=1)
    parser.add_argument("--fixed-track-count", type=int, default=0)
    parser.add_argument("--fixed-coop-count", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--allow-timestamp-output-fallback", action="store_true")
    parser.add_argument("--allow-negative-inf-output", action="append", default=[])
    parser.add_argument(
        "--diagnostic-ranges-jsonl",
        help="Optional per-frame tensor-range trace, including a failing frame.",
    )
    return parser.parse_args()


def scalarize(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return scalarize(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, np.ndarray):
        return scalarize(value.tolist())
    if isinstance(value, dict):
        return {key: scalarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalarize(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_progress(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "fps_from_mean": float(1000.0 / values.mean()),
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonfinite_tensors(outputs):
    return {
        name: int((~torch.isfinite(value)).sum().item())
        for name, value in outputs.items()
        if value.is_floating_point() and not torch.isfinite(value).all()
    }


def tensor_range(value):
    value = value.detach()
    if not value.is_floating_point():
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": int(value.min().item()) if value.numel() else None,
            "max": int(value.max().item()) if value.numel() else None,
            "nonfinite": 0,
        }
    finite = torch.isfinite(value)
    finite_values = value[finite]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
        "nonfinite": int((~finite).sum().item()),
    }


def selected_ranges(values, names):
    return {
        name: tensor_range(values[name])
        for name in names
        if name in values
    }


def tracking_state_statistics(outputs):
    obj_idxes = outputs.get("prev_track_intances3_out")
    scores = outputs.get("prev_track_intances7_out")
    if obj_idxes is None or scores is None:
        return {}
    return {
        "query_count": int(obj_idxes.numel()),
        "active_obj_idx_count": int((obj_idxes >= 0).sum().item()),
        "score_ge_0_05_count": int((scores >= 0.05).sum().item()),
        "score_ge_0_35_count": int((scores >= 0.35).sum().item()),
        "score_ge_0_40_count": int((scores >= 0.40).sum().item()),
        "score_min": float(scores.min().item()),
        "score_max": float(scores.max().item()),
    }


def consume_expected_negative_infinities(nonfinite, outputs, allowed_names):
    counts = {}
    for name in allowed_names:
        value = outputs.get(name)
        if value is None or name not in nonfinite:
            continue
        nan_count = int(torch.isnan(value).sum().item())
        positive_inf_count = int(torch.isposinf(value).sum().item())
        negative_inf_count = int(torch.isneginf(value).sum().item())
        if nan_count or positive_inf_count or negative_inf_count != nonfinite[name]:
            raise RuntimeError({
                "output": name,
                "nan": nan_count,
                "positive_inf": positive_inf_count,
                "negative_inf": negative_inf_count,
                "all_nonfinite": dict(nonfinite),
            })
        counts[name] = negative_inf_count
        nonfinite.pop(name)
    return counts


def iou_components(prediction, target):
    prediction = prediction.reshape(1, -1)
    target = target.reshape(1, -1)
    intersection = (prediction * target).sum(dim=1).cpu()
    union = (
        prediction.sum(dim=1) + target.sum(dim=1) - intersection.to(prediction)
    ).cpu()
    score = intersection / (union + 1.0e-13)
    return score, intersection, union


def map_metrics(outputs, ego_data):
    labels = agent_tensor(ego_data, "gt_lane_labels", outputs["lane_pred"].device, torch.int64)
    masks = agent_tensor(ego_data, "gt_lane_masks", outputs["lane_pred"].device, torch.int32)
    drivable_pred = outputs["drivable_pred"].to(torch.int32)
    lane_pred = outputs["lane_pred"].to(torch.int32)
    drivable_gt = masks[0, -1]
    lanes_pred = (lane_pred.sum(0) > 0).to(torch.int32)
    lanes_gt = (masks[0, :-1].sum(0) > 0).to(torch.int32)
    class_gt = [
        (masks[0][labels[0] == class_id].sum(0) > 0).to(torch.int32)
        for class_id in range(3)
    ]

    values = {}
    for name, prediction, target in (
        ("drivable", drivable_pred, drivable_gt),
        ("lanes", lanes_pred, lanes_gt),
        ("divider", lane_pred[0], class_gt[0]),
        ("crossing", lane_pred[1], class_gt[1]),
        ("contour", lane_pred[2], class_gt[2]),
    ):
        score, intersection, union = iou_components(prediction, target)
        values[name + "_iou"] = score
        values[name + "_intersection"] = intersection
        values[name + "_union"] = union
    return values, drivable_gt


def consecutive_instance_ground_truth(instance, ignore_index=255):
    result = torch.zeros_like(instance)
    next_id = 1
    for identifier in torch.unique(instance):
        if int(identifier.item()) in (0, ignore_index):
            continue
        result[instance == identifier] = next_id
        next_id += 1
    return result.long()


class PlanningPostprocessor:
    planning_steps = 10
    occ_n_future_only_occ = 4
    bev_h = 200
    bev_w = 200
    occ_filter_range = 5.0
    sigma = 1.0
    alpha_collision = 5.0

    collision_optimization = PlanningHeadSingleMode.collision_optimization
    drivable_optimization = PlanningHeadSingleMode.drivable_optimization

    def __call__(self, trajectory, occupancy, drivable):
        trajectory = self.collision_optimization(trajectory, occupancy)
        return self.drivable_optimization(trajectory, drivable)


def make_result(outputs, ego_data, map_result, planning_trajectory):
    meta = metadata(ego_data)
    box_type = meta["box_type_3d"]
    track_boxes = box_type(outputs["bboxes_dict_bboxes"].detach().cpu(), 9)
    track_scores = outputs["scores"].detach().cpu()
    track_labels = outputs["labels"].detach().cpu().long()
    bbox_index = outputs["bbox_index"].detach().cpu().long()
    track_ids = outputs["obj_idxes"].detach().cpu().long()
    track_mask = torch.ones_like(track_scores, dtype=torch.bool)

    det_boxes = box_type(outputs["det_bboxes"].detach().cpu(), 9)
    det_scores = outputs["det_scores"].detach().cpu()
    det_labels = outputs["det_labels"].detach().cpu().long()

    result = {
        "token": meta["sample_idx"],
        "track_bbox_results": [[
            track_boxes, track_scores, track_labels, bbox_index, track_mask
        ]],
        "boxes_3d": track_boxes,
        "scores_3d": track_scores,
        "labels_3d": track_labels,
        "track_scores": track_scores,
        "track_ids": track_ids,
        "boxes_3d_det": det_boxes,
        "scores_3d_det": det_scores,
        "labels_3d_det": det_labels,
        "ret_iou": {key: map_result[key] for key in MAP_RESULT_KEYS},
        "planning_traj": planning_trajectory.detach().cpu(),
        "planning_traj_gt": [agent_tensor(
            ego_data, "sdc_planning", torch.device("cpu"), torch.float64
        )],
        "command": [agent_tensor(
            ego_data, "command", torch.device("cpu"), torch.int64
        )],
    }

    active_count = track_labels.numel()
    labels_device = outputs["labels"]
    vehicle_mask = torch.zeros_like(labels_device, dtype=torch.bool)
    for class_id in VEHICLE_LABELS:
        vehicle_mask |= labels_device == class_id
    vehicle_indices = torch.where(vehicle_mask)[0]
    motion_indices = torch.cat([
        vehicle_indices,
        vehicle_indices.new_tensor([outputs["motion_trajs"].shape[2] - 1]),
    ])
    if active_count and vehicle_indices.numel() == 0:
        motion_indices = motion_indices[-1:]
    trajectories = outputs["motion_trajs"][:, 0, motion_indices].detach().cpu()
    trajectory_scores = outputs["motion_traj_scores"][:, 0, motion_indices].detach().cpu()

    # The motion head emits vehicle-class trajectories followed by the SDC,
    # while the dataset formatter indexes trajectories by every tracking box.
    # Preserve that SDC tail and insert zero placeholders for non-motion boxes.
    aligned_trajectories = trajectories.new_zeros(
        (trajectories.shape[0], active_count + 1, *trajectories.shape[2:])
    )
    aligned_trajectory_scores = trajectory_scores.new_zeros(
        (trajectory_scores.shape[0], active_count + 1, *trajectory_scores.shape[2:])
    )
    vehicle_indices_cpu = vehicle_indices.detach().cpu()
    vehicle_count = vehicle_indices_cpu.numel()
    if vehicle_count:
        aligned_trajectories[:, vehicle_indices_cpu] = trajectories[:, :vehicle_count]
        aligned_trajectory_scores[:, vehicle_indices_cpu] = trajectory_scores[:, :vehicle_count]
    aligned_trajectories[:, -1] = trajectories[:, -1]
    aligned_trajectory_scores[:, -1] = trajectory_scores[:, -1]
    trajectories = aligned_trajectories
    trajectory_scores = aligned_trajectory_scores
    result.update({
        "traj_0": trajectories[0],
        "traj_scores_0": trajectory_scores[0],
        "traj_1": trajectories[1],
        "traj_scores_1": trajectory_scores[1],
        "traj": trajectories[-1],
        "traj_scores": trajectory_scores[-1],
    })
    return result


def filtered_inputs(engine, values):
    return {name: values[name] for name in engine.input_names}


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.output_dir)
    if args.diagnostic_ranges_jsonl:
        mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.diagnostic_ranges_jsonl)))
        open(args.diagnostic_ranges_jsonl, "w").close()
    cfg = load_config(args.config, workers=args.workers)
    patch_disk_backend_file_objects()
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    infrastructure_engine = TensorRTEngine(
        args.infrastructure_engine,
        args.plugin,
        context_cache_size=args.context_cache_size,
    )
    ego_engine = TensorRTEngine(
        args.ego_engine,
        args.plugin,
        context_cache_size=args.context_cache_size,
    )
    device = torch.device("cuda")
    infrastructure_state = AgentState(device, args.fixed_track_count)
    ego_state = AgentState(device, args.fixed_track_count)
    execution_stream = torch.cuda.Stream()

    ranges = {"30x30": (70, 130), "100x100": (0, 200)}
    iou_metrics = {key: IntersectionOverUnion(2).cuda() for key in ranges}
    panoptic_metrics = {
        key: PanopticMetric(n_classes=2, temporally_consistent=True).cuda()
        for key in ranges
    }
    planning_metric = PlanningMetric(n_future=dataset.planning_steps).cuda()
    planning_postprocess = PlanningPostprocessor()

    frame_limit = len(dataset) if args.max_frames <= 0 else min(args.max_frames, len(dataset))
    progress_path = osp.join(args.output_dir, "progress.json")
    run_started = time.perf_counter()
    write_progress(progress_path, {
        "status": "running",
        "completed_frames": 0,
        "total_frames": frame_limit,
        "elapsed_seconds": 0.0,
    })
    iterator = iter(loader)
    progress = mmcv.ProgressBar(frame_limit)
    results = []
    rows = []
    cooperative_rows = []
    num_occ = 0
    timestamp_fallbacks = {"infrastructure": 0, "ego": 0}
    allowed_negative_inf_counts = {}

    for frame_index in range(frame_limit):
        e2e_start = time.perf_counter()
        data_start = e2e_start
        data = next(iterator)
        data_end = time.perf_counter()
        ego_data = data["ego_agent_data"]
        infrastructure_data = data["other_agent_data_dict"]["model_other_agent_inf"]

        with torch.cuda.stream(execution_stream):
            infrastructure_inputs, infrastructure_new_scene = (
                infrastructure_state.build_inputs(infrastructure_data)
            )
            execution_stream.synchronize()
            infrastructure_start = time.perf_counter()
            infrastructure_outputs = infrastructure_engine.infer(
                filtered_inputs(infrastructure_engine, infrastructure_inputs),
                synchronize=False,
            )
            execution_stream.synchronize()
            infrastructure_end = time.perf_counter()
            infrastructure_nonfinite = nonfinite_tensors(infrastructure_outputs)
            for name, count in consume_expected_negative_infinities(
                infrastructure_nonfinite,
                infrastructure_outputs,
                args.allow_negative_inf_output,
            ).items():
                allowed_negative_inf_counts[name] = (
                    allowed_negative_inf_counts.get(name, 0) + count
                )
            infrastructure_timestamp_fallback = False
            if (
                args.allow_timestamp_output_fallback
                and set(infrastructure_nonfinite) == {"prev_timestamp_out"}
                and not nonfinite_tensors({"timestamp": infrastructure_inputs["timestamp"]})
            ):
                infrastructure_nonfinite.pop("prev_timestamp_out")
                infrastructure_timestamp_fallback = True
                timestamp_fallbacks["infrastructure"] += 1
            if infrastructure_nonfinite:
                raise RuntimeError({
                    "frame": frame_index,
                    "agent": "infrastructure",
                    "nonfinite": infrastructure_nonfinite,
                })
            infrastructure_state.update(
                infrastructure_outputs,
                timestamp_fallback=infrastructure_inputs["timestamp"]
                if infrastructure_timestamp_fallback else None,
            )

            ego_inputs, ego_new_scene = ego_state.build_inputs(ego_data)
            veh2inf_rt = agent_tensor(ego_data, "veh2inf_rt", device, torch.float32)
            cooperative_inputs, cooperative_stats = prepare_cooperative_inputs(
                infrastructure_outputs,
                ego_state,
                veh2inf_rt,
                fixed_coop_count=args.fixed_coop_count,
            )
            ego_inputs.update(cooperative_inputs)
            execution_stream.synchronize()
            ego_start = time.perf_counter()
            ego_outputs = ego_engine.infer(
                filtered_inputs(ego_engine, ego_inputs), synchronize=False
            )
            execution_stream.synchronize()
            ego_end = time.perf_counter()
            if args.diagnostic_ranges_jsonl:
                diagnostic = {
                    "frame": frame_index,
                    "infrastructure_new_scene": infrastructure_new_scene,
                    "ego_new_scene": ego_new_scene,
                    "infrastructure_scene_token": str(
                        metadata(infrastructure_data)["scene_token"]
                    ),
                    "ego_scene_token": str(metadata(ego_data)["scene_token"]),
                    "infrastructure_inputs": selected_ranges(
                        infrastructure_inputs,
                        ("prev_bev", "timestamp", "prev_timestamp", "use_prev_bev"),
                    ),
                    "infrastructure_outputs": selected_ranges(
                        infrastructure_outputs,
                        (
                            "bev_embed", "lane_reference", "prev_timestamp_out",
                            "prev_track_intances3_out", "prev_track_intances7_out",
                            "scores", "max_obj_id_out",
                        ),
                    ),
                    "infrastructure_tracking": tracking_state_statistics(
                        infrastructure_outputs
                    ),
                    "ego_inputs": selected_ranges(
                        ego_inputs,
                        (
                            "prev_bev", "timestamp", "prev_timestamp", "use_prev_bev",
                            "coop_lane_reference", "coop_occ_pred_scores",
                        ),
                    ),
                    "ego_outputs": selected_ranges(
                        ego_outputs,
                        (
                            "bev_embed", "lane_reference", "prev_timestamp_out",
                            "value.31", "attention_weights.31", "sampling_offsets.31",
                            "onnx::Shape_6601", "onnx::ReduceMean_6609",
                            "onnx::MatMul_6610", "onnx::Add_6612", "input.959",
                            "bev_query.3", "bev_query.7", "bev_query.11",
                            "bev_query.15", "bev_query.19",
                            "bev_embed.1", "input.1011", "added_query",
                            "onnx::Gemm_7152", "onnx::Slice_7221",
                            "onnx::Add_7234", "bev_embed.3",
                            "occ_pred_sigmoid", "outs_planning",
                        ),
                    ),
                    "ego_tracking": tracking_state_statistics(ego_outputs),
                    "cooperative_stats": cooperative_stats,
                }
                with open(args.diagnostic_ranges_jsonl, "a") as handle:
                    handle.write(json.dumps(diagnostic, allow_nan=False) + "\n")
            ego_nonfinite = nonfinite_tensors(ego_outputs)
            for name, count in consume_expected_negative_infinities(
                ego_nonfinite, ego_outputs, args.allow_negative_inf_output
            ).items():
                allowed_negative_inf_counts[name] = (
                    allowed_negative_inf_counts.get(name, 0) + count
                )
            ego_timestamp_fallback = False
            if (
                args.allow_timestamp_output_fallback
                and set(ego_nonfinite) == {"prev_timestamp_out"}
                and not nonfinite_tensors({"timestamp": ego_inputs["timestamp"]})
            ):
                ego_nonfinite.pop("prev_timestamp_out")
                ego_timestamp_fallback = True
                timestamp_fallbacks["ego"] += 1
            if ego_nonfinite:
                raise RuntimeError({
                    "frame": frame_index,
                    "agent": "ego",
                    "nonfinite": ego_nonfinite,
                })
            ego_state.update(
                ego_outputs,
                timestamp_fallback=ego_inputs["timestamp"]
                if ego_timestamp_fallback else None,
            )

        map_result, drivable_gt = map_metrics(ego_outputs, ego_data)
        occupancy = ego_outputs["seg_out"].long()
        instance_prediction = predict_instance_segmentation_and_trajectories(
            occupancy, ego_outputs["occ_pred_sigmoid"]
        )
        instance_gt = consecutive_instance_ground_truth(
            agent_tensor(ego_data, "gt_instance", device, torch.int64)[:, :5]
        )
        segmentation_gt = agent_tensor(
            ego_data, "gt_segmentation", device, torch.int64
        )[:, :5, None]
        invalid_occ = bool(agent_tensor(
            ego_data, "gt_occ_has_invalid_frame", device, torch.bool
        ).item())
        if not invalid_occ:
            num_occ += 1
            for key, (start, end) in ranges.items():
                limits = slice(start, end)
                iou_metrics[key](
                    occupancy[..., limits, limits].contiguous(),
                    segmentation_gt[..., limits, limits].contiguous(),
                )
                panoptic_metrics[key](
                    instance_prediction[..., limits, limits].contiguous(),
                    instance_gt[..., limits, limits].contiguous(),
                )

        planning_trajectory = planning_postprocess(
            ego_outputs["outs_planning"], occupancy, drivable_gt
        )
        planning_gt = agent_tensor(ego_data, "sdc_planning", device, torch.float32)
        planning_mask = agent_tensor(
            ego_data, "sdc_planning_mask", device, torch.float32
        )
        segmentation_full = agent_tensor(
            ego_data, "gt_segmentation", device, torch.int64
        )
        steps = dataset.planning_steps
        planning_metric(
            planning_trajectory[:, :steps, :2].clone(),
            planning_gt[0, :, :steps, :2].clone(),
            planning_mask[0, :, :steps, :2].clone(),
            segmentation_full[:, 1:steps + 1],
            drivable_gt,
        )
        results.append(make_result(
            ego_outputs, ego_data, map_result, planning_trajectory
        ))
        e2e_end = time.perf_counter()

        rows.append({
            "frame": frame_index,
            "data_ms": (data_end - data_start) * 1000.0,
            "infrastructure_forward_ms": (infrastructure_end - infrastructure_start) * 1000.0,
            "ego_forward_ms": (ego_end - ego_start) * 1000.0,
            "forward_ms": (ego_end - infrastructure_start) * 1000.0,
            "end_to_end_ms": (e2e_end - e2e_start) * 1000.0,
        })
        cooperative_rows.append({"frame": frame_index, **cooperative_stats})
        cooperative_rows[-1].update({
            "infrastructure_timestamp_fallback": int(infrastructure_timestamp_fallback),
            "ego_timestamp_fallback": int(ego_timestamp_fallback),
            "infrastructure_prev_track_count": int(
                infrastructure_inputs["prev_track_intances0"].shape[0]
            ),
            "ego_prev_track_count": int(
                ego_inputs["prev_track_intances0"].shape[0]
            ),
        })
        progress.update()
        completed_frames = frame_index + 1
        progress_interval = max(1, args.progress_interval)
        if completed_frames % progress_interval == 0 or completed_frames == frame_limit:
            write_progress(progress_path, {
                "status": "running",
                "completed_frames": completed_frames,
                "total_frames": frame_limit,
                "elapsed_seconds": time.perf_counter() - run_started,
                "last_frame": rows[-1],
                "runtime_output_buffers": {
                    "infrastructure": infrastructure_engine.allocation_stats(),
                    "ego": ego_engine.allocation_stats(),
                },
                "gpu_memory": {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                },
            })

    write_progress(progress_path, {
        "status": "inference_complete",
        "completed_frames": frame_limit,
        "total_frames": frame_limit,
        "elapsed_seconds": time.perf_counter() - run_started,
    })

    occupancy_result = {}
    for key in ranges:
        panoptic = panoptic_metrics[key].compute()
        for metric_name, value in panoptic.items():
            occupancy_result.setdefault(metric_name, []).append(100 * value[1].item())
        occupancy_result.setdefault("iou", []).append(
            100 * iou_metrics[key].compute()[1].item()
        )
    occupancy_result["num_occ"] = num_occ
    occupancy_result["ratio_occ"] = num_occ / frame_limit
    planning_result = planning_metric.compute()
    result_bundle = {
        "bbox_results": results,
        "occ_results_computed": occupancy_result,
        "planning_results_computed": planning_result,
    }
    mmcv.dump(result_bundle, osp.join(args.output_dir, "results.pkl"))
    with open(osp.join(args.output_dir, "task_metrics.json"), "w") as handle:
        json.dump({
            "occupancy": scalarize(occupancy_result),
            "planning": scalarize(planning_result),
        }, handle, indent=2, allow_nan=False)
    with open(osp.join(args.output_dir, "latency_frames.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(osp.join(args.output_dir, "cooperative_frames.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cooperative_rows[0].keys())
        writer.writeheader()
        writer.writerows(cooperative_rows)

    warmup = min(args.warmup_frames, max(0, len(rows) - 1))
    measured = rows[warmup:]
    latency = {
        "schema_version": 1,
        "framework": "TensorRT",
        "precision": args.precision,
        "gpu": torch.cuda.get_device_name(0),
        "config": os.path.abspath(args.config),
        "engines": {
            "infrastructure": {
                "path": os.path.abspath(args.infrastructure_engine),
                "sha256": sha256(args.infrastructure_engine),
            },
            "ego": {
                "path": os.path.abspath(args.ego_engine),
                "sha256": sha256(args.ego_engine),
            },
            "plugin": {
                "path": os.path.abspath(args.plugin),
                "sha256": sha256(args.plugin),
            },
        },
        "dataset_frames": frame_limit,
        "warmup_frames_excluded": warmup,
        "fixed_input_shapes": {
            "track_count": args.fixed_track_count or None,
            "cooperative_track_count": args.fixed_coop_count or None,
            "padding_semantics": (
                "Temporal rows use the graph's -10000 invalid sentinel; "
                "cooperative rows use an out-of-range matched index and are "
                "excluded from fusion."
            ),
        },
        "runtime_fallbacks": {
            "timestamp_identity_output_to_input_enabled": args.allow_timestamp_output_fallback,
            "counts": timestamp_fallbacks,
            "semantic_basis": "prev_timestamp_out is ONNX Identity(timestamp)",
            "allowed_negative_inf_output_counts": allowed_negative_inf_counts,
        },
        "runtime_output_buffers": {
            "infrastructure": infrastructure_engine.allocation_stats(),
            "ego": ego_engine.allocation_stats(),
        },
        "definitions": {
            "forward": "Infrastructure engine, cooperative host matching, and ego engine with CUDA synchronization",
            "end_to_end": "Dataloader wait, forward, official occupancy/planning updates, result reconstruction, and planning postprocess",
        },
        "infrastructure_forward": summarize([
            row["infrastructure_forward_ms"] for row in measured
        ]),
        "ego_forward": summarize([row["ego_forward_ms"] for row in measured]),
        "forward": summarize([row["forward_ms"] for row in measured]),
        "end_to_end": summarize([row["end_to_end_ms"] for row in measured]),
    }
    with open(osp.join(args.output_dir, "latency_metrics.json"), "w") as handle:
        json.dump(latency, handle, indent=2, allow_nan=False)

    if args.evaluate and frame_limit == len(dataset):
        eval_dir = osp.join(args.output_dir, "full_task_eval")
        mmcv.mkdir_or_exist(eval_dir)
        detail = dataset.evaluate(
            result_bundle,
            metric=["bbox"],
            jsonfile_prefix=eval_dir,
            pipeline=cfg.get("evaluation", {}).get("pipeline"),
        )
        with open(osp.join(args.output_dir, "dataset_metrics.json"), "w") as handle:
            json.dump(scalarize(detail), handle, indent=2, allow_nan=True)

    write_progress(progress_path, {
        "status": "complete",
        "completed_frames": frame_limit,
        "total_frames": frame_limit,
        "elapsed_seconds": time.perf_counter() - run_started,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure_args = parse_args()
        failure_progress = osp.join(failure_args.output_dir, "progress.json")
        previous = {}
        if osp.exists(failure_progress):
            with open(failure_progress) as handle:
                previous = json.load(handle)
        previous.update({
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        mmcv.mkdir_or_exist(failure_args.output_dir)
        write_progress(failure_progress, previous)
        raise
