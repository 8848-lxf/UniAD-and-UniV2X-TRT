#!/usr/bin/env python3
"""Replay a synchronized CARLA camera sequence through UniV2X TensorRT."""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from cooperative_runtime import prepare_cooperative_inputs
from trt_engine import TensorRTEngine
from trt_runtime import INPUT_NAMES, TRACK_OUTPUT_NAMES, empty_track_state


MEAN_BGR = np.asarray([103.530, 116.280, 123.675], dtype=np.float32)
HAND_FLIP = np.diag([1.0, -1.0, 1.0, 1.0])
CAMERA_RH_TO_STANDARD = np.asarray([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    parser.add_argument("infrastructure_engine")
    parser.add_argument("ego_engine")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--template-input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--context-cache-size", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--match-distance", type=float, default=2.0)
    parser.add_argument("--fixed-track-count", type=int, default=0)
    parser.add_argument("--fixed-coop-count", type=int, default=0)
    parser.add_argument("--allow-timestamp-output-fallback", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def right_handed(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return HAND_FLIP.dot(matrix).dot(HAND_FLIP)


def intrinsic(width, height, fov_degrees):
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    value = np.eye(4, dtype=np.float64)
    value[0, 0] = focal
    value[1, 1] = focal
    value[0, 2] = width / 2.0
    value[1, 2] = height / 2.0
    return value


def lidar2image(world_from_agent, world_from_camera, camera):
    world_from_agent = right_handed(world_from_agent)
    world_from_camera = right_handed(world_from_camera)
    camera_from_agent = np.linalg.inv(world_from_camera).dot(world_from_agent)
    projection = intrinsic(
        camera["width"], camera["height"], camera["fov_degrees"]
    ).dot(CAMERA_RH_TO_STANDARD).dot(camera_from_agent)
    return projection.astype(np.float32)


def model_image(bgr):
    if bgr.shape != (1080, 1920, 3):
        raise ValueError("unexpected CARLA image shape: %r" % (bgr.shape,))
    normalized = bgr.astype(np.float32) - MEAN_BGR
    padded = np.zeros((1088, 1920, 3), dtype=np.float32)
    padded[:1080] = normalized
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None, None])


def pose_inputs(world_from_agent, velocity=None, acceleration=None, angular=None):
    transform = right_handed(world_from_agent)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[:3] = translation
    can_bus[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    if acceleration is not None:
        can_bus[7:10] = [acceleration[0], -acceleration[1], acceleration[2]]
    if angular is not None:
        can_bus[10:13] = [angular[0], -angular[1], angular[2]]
    if velocity is not None:
        can_bus[13:16] = [velocity[0], -velocity[1], velocity[2]]
    can_bus[-2] = yaw
    can_bus[-1] = math.degrees(yaw)
    return (
        rotation.T.astype(np.float32)[None],
        translation.astype(np.float32)[None],
        can_bus,
    )


class ReplayAgentState:
    def __init__(self, device, templates, scene_token, fixed_track_count=0):
        self.device = device
        self.templates = templates
        self.fixed_track_count = fixed_track_count
        encoded = scene_token.encode("utf-8")[:32]
        self.scene_bytes = np.zeros(32, dtype=np.uint8)
        self.scene_bytes[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        self.reset()

    def reset(self):
        self.tracks = empty_track_state(self.device)
        self.prev_timestamp = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.timestamp_origin = None
        self.prev_l2g_r_mat = torch.zeros(1, 3, 3, dtype=torch.float32, device=self.device)
        self.prev_l2g_t = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.prev_bev = torch.zeros(40000, 1, 256, dtype=torch.float32, device=self.device)
        self.max_obj_id = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.prev_position = None
        self.prev_angle = None
        self.frame_index = 0

    def build_inputs(self, image, timestamp, l2g_r, l2g_t, can_bus, lidar2img):
        relative_can_bus = can_bus.copy()
        if self.prev_position is None:
            relative_can_bus[:3] = 0.0
            relative_can_bus[-1] = 0.0
        else:
            relative_can_bus[:3] -= self.prev_position
            relative_can_bus[-1] -= self.prev_angle
        self.prev_position = can_bus[:3].copy()
        self.prev_angle = float(can_bus[-1])
        if self.timestamp_origin is None:
            self.timestamp_origin = timestamp
        relative_timestamp = timestamp - self.timestamp_origin
        input_tracks = self.tracks
        if self.fixed_track_count:
            current = int(input_tracks[0].shape[0])
            if current > self.fixed_track_count:
                raise RuntimeError(
                    "track count %d exceeds fixed input capacity %d"
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
            self.templates["gt_lane_labels"],
            self.templates["gt_lane_masks"],
            self.templates["gt_segmentation"],
            torch.from_numpy(self.scene_bytes).to(self.device),
            torch.tensor(
                [relative_timestamp], dtype=torch.float32, device=self.device
            ),
            torch.from_numpy(l2g_r).to(self.device),
            torch.from_numpy(l2g_t).to(self.device),
            torch.from_numpy(image).to(self.device),
            torch.from_numpy(relative_can_bus).to(self.device),
            torch.from_numpy(np.repeat(lidar2img[None, None], 6, axis=1)).to(self.device),
            torch.tensor([1088.0, 1920.0], dtype=torch.float32, device=self.device),
            torch.tensor([2], dtype=torch.int64, device=self.device),
            torch.tensor([int(self.frame_index > 0)], dtype=torch.int32, device=self.device),
            self.max_obj_id,
        ]
        self.frame_index += 1
        return dict(zip(INPUT_NAMES, values))

    def update(self, outputs, timestamp_fallback=None):
        self.tracks = [outputs[name] for name in TRACK_OUTPUT_NAMES]
        timestamp = outputs["prev_timestamp_out"]
        if not torch.isfinite(timestamp).all():
            if timestamp_fallback is None:
                raise RuntimeError("non-finite prev_timestamp_out")
            timestamp = timestamp_fallback
        self.prev_timestamp = timestamp
        self.prev_l2g_t = outputs["prev_l2g_t_out"]
        self.prev_l2g_r_mat = outputs["prev_l2g_r_mat_out"]
        self.prev_bev = outputs["bev_embed"]
        self.max_obj_id = outputs["max_obj_id_out"]


def load_templates(directory, device):
    result = {}
    for side in ("ego", "infrastructure"):
        arrays = np.load(os.path.join(directory, side + ".npz"))
        result[side] = {
            "gt_lane_labels": torch.zeros_like(
                torch.from_numpy(arrays["gt_lane_labels"]), device=device
            ),
            "gt_lane_masks": torch.zeros_like(
                torch.from_numpy(arrays["gt_lane_masks"]), device=device
            ),
            "gt_segmentation": torch.zeros_like(
                torch.from_numpy(arrays["gt_segmentation"]), device=device
            ),
        }
    return result


def filtered_inputs(engine, values):
    return {name: values[name] for name in engine.input_names}


def nonfinite(outputs):
    return [
        name for name, value in outputs.items()
        if value.is_floating_point() and not torch.isfinite(value).all()
    ]


def vehicle_centers_in_ego(record, ego_actor_id, world_from_ego):
    ego_from_world = np.linalg.inv(right_handed(world_from_ego))
    centers = []
    for vehicle in record["vehicles"]:
        if int(vehicle["id"]) == int(ego_actor_id):
            continue
        center_world = right_handed(vehicle["world_from_vehicle"])[:, 3]
        center = ego_from_world.dot(center_world)[:3]
        if abs(center[0]) <= 51.2 and abs(center[1]) <= 51.2:
            centers.append(center)
    return np.asarray(centers, dtype=np.float32).reshape(-1, 3)


def match_centers(predicted, ground_truth, max_distance):
    if not len(predicted) or not len(ground_truth):
        return 0
    costs = np.linalg.norm(
        predicted[:, None, :2] - ground_truth[None, :, :2], axis=-1
    )
    rows, columns = linear_sum_assignment(costs)
    return int((costs[rows, columns] <= max_distance).sum())


def planning_l2(rows, transforms):
    horizons = [[] for _ in range(10)]
    for index, row in enumerate(rows):
        ego_from_world = np.linalg.inv(right_handed(transforms[index]))
        prediction = np.asarray(row["planning_xy"], dtype=np.float64)
        for horizon in range(min(10, len(transforms) - index - 1)):
            future_world = right_handed(transforms[index + horizon + 1])[:, 3]
            target = ego_from_world.dot(future_world)[:2]
            horizons[horizon].append(float(np.linalg.norm(prediction[horizon] - target)))
    return [float(np.mean(values)) if values else None for values in horizons]


def main():
    args = parse_args()
    with open(os.path.join(args.capture_dir, "manifest.json")) as handle:
        manifest = json.load(handle)
    device = torch.device("cuda")
    templates = load_templates(args.template_input_dir, device)
    infrastructure_engine = TensorRTEngine(
        args.infrastructure_engine, args.plugin,
        context_cache_size=args.context_cache_size,
    )
    ego_engine = TensorRTEngine(
        args.ego_engine, args.plugin,
        context_cache_size=args.context_cache_size,
    )
    scene_token = "carla_%s" % manifest["map"]
    infrastructure_state = ReplayAgentState(
        device, templates["infrastructure"], scene_token, args.fixed_track_count
    )
    ego_state = ReplayAgentState(
        device, templates["ego"], scene_token, args.fixed_track_count
    )
    camera = manifest["capture"]["camera"]
    execution_stream = torch.cuda.Stream()
    rows = []
    ego_transforms = []
    total_matches = 0
    total_predictions = 0
    total_ground_truth = 0
    timestamp_fallbacks = {"infrastructure": 0, "ego": 0}

    for record in manifest["frames"]:
        e2e_start = time.perf_counter()
        arrays = np.load(os.path.join(args.capture_dir, record["file"]))
        ego_image = model_image(arrays["ego_bgr"])
        infrastructure_image = model_image(arrays["infrastructure_bgr"])
        timestamp = float(arrays["timestamp"][0])
        ego_l2g_r, ego_l2g_t, ego_can_bus = pose_inputs(
            arrays["world_from_ego"], arrays["ego_velocity"],
            arrays["ego_acceleration"], arrays["ego_angular_velocity"],
        )
        infrastructure_l2g_r, infrastructure_l2g_t, infrastructure_can_bus = pose_inputs(
            arrays["world_from_infrastructure"]
        )
        ego_projection = lidar2image(
            arrays["world_from_ego"], arrays["world_from_ego_camera"], camera
        )
        infrastructure_projection = lidar2image(
            arrays["world_from_infrastructure"],
            arrays["world_from_infrastructure_camera"], camera,
        )
        with torch.cuda.stream(execution_stream):
            infrastructure_inputs = infrastructure_state.build_inputs(
                infrastructure_image, timestamp, infrastructure_l2g_r,
                infrastructure_l2g_t, infrastructure_can_bus,
                infrastructure_projection,
            )
            execution_stream.synchronize()
            infrastructure_start = time.perf_counter()
            infrastructure_outputs = infrastructure_engine.infer(
                filtered_inputs(infrastructure_engine, infrastructure_inputs),
                synchronize=False,
            )
            execution_stream.synchronize()
            infrastructure_end = time.perf_counter()
            bad = nonfinite(infrastructure_outputs)
            infrastructure_fallback = bad == ["prev_timestamp_out"]
            if bad and not (
                infrastructure_fallback and args.allow_timestamp_output_fallback
            ):
                raise RuntimeError("non-finite infrastructure outputs: %r" % bad)
            if infrastructure_fallback:
                timestamp_fallbacks["infrastructure"] += 1
            infrastructure_state.update(
                infrastructure_outputs,
                infrastructure_inputs["timestamp"] if infrastructure_fallback else None,
            )

            ego_inputs = ego_state.build_inputs(
                ego_image, timestamp, ego_l2g_r, ego_l2g_t, ego_can_bus,
                ego_projection,
            )
            world_from_ego = right_handed(arrays["world_from_ego"])
            world_from_infrastructure = right_handed(
                arrays["world_from_infrastructure"]
            )
            infrastructure_from_ego = np.linalg.inv(
                world_from_infrastructure
            ).dot(world_from_ego).T.astype(np.float32)[None]
            cooperative, cooperative_stats = prepare_cooperative_inputs(
                infrastructure_outputs,
                ego_state,
                torch.from_numpy(infrastructure_from_ego).to(device),
                fixed_coop_count=args.fixed_coop_count,
            )
            ego_inputs.update(cooperative)
            execution_stream.synchronize()
            ego_start = time.perf_counter()
            ego_outputs = ego_engine.infer(
                filtered_inputs(ego_engine, ego_inputs), synchronize=False
            )
            execution_stream.synchronize()
            ego_end = time.perf_counter()
            bad = nonfinite(ego_outputs)
            ego_fallback = bad == ["prev_timestamp_out"]
            if bad and not (ego_fallback and args.allow_timestamp_output_fallback):
                raise RuntimeError("non-finite ego outputs: %r" % bad)
            if ego_fallback:
                timestamp_fallbacks["ego"] += 1
            ego_state.update(
                ego_outputs, ego_inputs["timestamp"] if ego_fallback else None
            )

        scores = ego_outputs["det_scores"].detach().float().cpu().numpy()
        boxes = ego_outputs["det_bboxes"].detach().float().cpu().numpy()
        keep = scores >= args.score_threshold
        predictions = boxes[keep, :3]
        ground_truth = vehicle_centers_in_ego(
            record, manifest["ego_actor_id"], arrays["world_from_ego"]
        )
        matches = match_centers(predictions, ground_truth, args.match_distance)
        total_matches += matches
        total_predictions += len(predictions)
        total_ground_truth += len(ground_truth)
        e2e_end = time.perf_counter()
        planning = ego_outputs["outs_planning"].detach().float().cpu().numpy()[0]
        rows.append({
            "frame": int(record["index"]),
            "infrastructure_forward_ms": (
                infrastructure_end - infrastructure_start
            ) * 1000.0,
            "ego_forward_ms": (ego_end - ego_start) * 1000.0,
            "forward_ms": (ego_end - infrastructure_start) * 1000.0,
            "end_to_end_ms": (e2e_end - e2e_start) * 1000.0,
            "prediction_count": int(len(predictions)),
            "ground_truth_count": int(len(ground_truth)),
            "matched_count": matches,
            "infrastructure_timestamp_fallback": int(infrastructure_fallback),
            "ego_timestamp_fallback": int(ego_fallback),
            "planning_xy": planning[:, :2].tolist(),
            "cooperative": cooperative_stats,
        })
        ego_transforms.append(np.asarray(arrays["world_from_ego"]))
        print("replayed %d/%d" % (len(rows), len(manifest["frames"])), flush=True)

    warmup = min(args.warmup_frames, max(0, len(rows) - 1))
    measured = rows[warmup:]
    result = {
        "schema": "univ2x_carla_open_loop_replay_v1",
        "precision": args.precision,
        "scope": (
            "CARLA-domain synchronized open-loop replay; detection center matching "
            "and planning consistency are diagnostic and are not SPD benchmark metrics"
        ),
        "capture": os.path.abspath(args.capture_dir),
        "frames": len(rows),
        "warmup_frames_excluded": warmup,
        "fixed_input_shapes": {
            "track_count": args.fixed_track_count or None,
            "cooperative_track_count": args.fixed_coop_count or None,
            "padding_semantics": (
                "Temporal rows use the graph's -10000 invalid sentinel; "
                "cooperative rows use an out-of-range matched index."
            ),
        },
        "runtime_fallbacks": {
            "timestamp_identity_output_to_input_enabled": (
                args.allow_timestamp_output_fallback
            ),
            "counts": timestamp_fallbacks,
        },
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
        "latency": {
            key: summarize([row[key + "_ms"] for row in measured])
            for key in (
                "infrastructure_forward", "ego_forward", "forward", "end_to_end"
            )
        },
        "detection_center_match": {
            "score_threshold": args.score_threshold,
            "match_distance_m": args.match_distance,
            "predictions": total_predictions,
            "ground_truth": total_ground_truth,
            "matches": total_matches,
            "precision": total_matches / max(1, total_predictions),
            "recall": total_matches / max(1, total_ground_truth),
        },
        "planning_l2_to_carla_autopilot_by_horizon_m": planning_l2(
            rows, ego_transforms
        ),
        "carla_events": manifest["events"],
        "runtime_output_buffers": {
            "infrastructure": infrastructure_engine.allocation_stats(),
            "ego": ego_engine.allocation_stats(),
        },
        "frames_detail": rows,
    }
    temporary = args.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
