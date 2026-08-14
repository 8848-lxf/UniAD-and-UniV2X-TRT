#!/usr/bin/env python3

"""Sweep occupancy thresholds with the deployment collision optimizer."""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
UNIAD_TOOLS = REPO_ROOT / "UniAD" / "repro" / "UniAD_deploy" / "tools"
sys.path.insert(0, str(UNIAD_TOOLS))
sys.path.insert(0, str(REPO_ROOT))

from evaluate_planning_outputs import (  # noqa: E402
    N_FUTURE,
    build_grid,
    collision_metrics,
    ego_footprint_pixels,
    load_ground_truth,
    load_predictions,
)
from UniV2X.deploy_int8.tools.planning_postprocess import (  # noqa: E402
    optimize_collision,
)


RAW_PREDICTIONS = None
SCORES = None
TARGETS = None
MASKS = None
SEGMENTATIONS = None
DX = None
BX = None
BEV_DIMENSION = None
EGO_FOOTPRINT = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--scores-manifest", required=True, type=Path)
    parser.add_argument("--raw-predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.075,0.09,0.10,0.11,0.125,0.15,0.20",
        help="Comma-separated scalar thresholds applied to every horizon",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def evaluate_threshold(threshold):
    frame_count = RAW_PREDICTIONS.shape[0]
    l2_sum = np.zeros(N_FUTURE, dtype=np.float64)
    point_collision_sum = np.zeros(N_FUTURE, dtype=np.float64)
    box_collision_sum = np.zeros(N_FUTURE, dtype=np.float64)
    modified_frames = 0
    modified_points = 0
    positive_cells = 0

    for frame in range(frame_count):
        occupancy = SCORES[frame] > threshold
        positive_cells += int(np.count_nonzero(occupancy))
        optimized, _ = optimize_collision(
            RAW_PREDICTIONS[frame], occupancy
        )
        delta = np.linalg.norm(
            optimized - RAW_PREDICTIONS[frame], axis=-1
        )
        changed = delta > 1.0e-7
        modified_frames += int(np.any(changed))
        modified_points += int(np.count_nonzero(changed))

        l2_sum += np.sqrt(
            np.sum(
                (optimized - TARGETS[frame]) ** 2 * MASKS[frame], axis=-1
            )
        )
        point_collision, box_collision = collision_metrics(
            optimized,
            TARGETS[frame],
            SEGMENTATIONS[frame],
            DX,
            BX,
            BEV_DIMENSION,
            EGO_FOOTPRINT,
        )
        point_collision_sum += point_collision
        box_collision_sum += box_collision

    l2 = l2_sum / frame_count
    point_collision = point_collision_sum / frame_count * 100.0
    box_collision = box_collision_sum / frame_count * 100.0
    return {
        "threshold": threshold,
        "frames": frame_count,
        "avg_l2_m": float(np.mean(l2)),
        "l2_by_timestep_m": l2.tolist(),
        "avg_point_collision_percent": float(np.mean(point_collision)),
        "point_collision_by_timestep_percent": point_collision.tolist(),
        "avg_box_collision_percent": float(np.mean(box_collision)),
        "box_collision_by_timestep_percent": box_collision.tolist(),
        "positive_occupancy_cells": positive_cells,
        "collision_optimizer_modified_frames": modified_frames,
        "collision_optimizer_modified_points": modified_points,
    }


def main():
    global RAW_PREDICTIONS, SCORES, TARGETS, MASKS, SEGMENTATIONS
    global DX, BX, BEV_DIMENSION, EGO_FOOTPRINT

    args = parse_args()
    thresholds = [
        float(item) for item in args.thresholds.split(",") if item.strip()
    ]
    if not thresholds:
        raise ValueError("No thresholds were specified")
    with args.scores_manifest.open() as handle:
        manifest = json.load(handle)
    raw_manifest_path = Path(str(args.raw_predictions) + ".manifest.json")
    raw_protocol = None
    if raw_manifest_path.is_file():
        with raw_manifest_path.open() as handle:
            raw_protocol = json.load(handle).get("temporal_protocol")
    score_protocol = manifest.get("temporal_protocol")
    if (raw_protocol is not None and score_protocol is not None
            and raw_protocol != score_protocol):
        raise ValueError(
            "Temporal protocol mismatch: occupancy scores use "
            f"{score_protocol}, raw predictions use {raw_protocol}"
        )
    frame_count = int(manifest["frames"])
    shape = tuple(int(value) for value in manifest["shape_per_frame"])
    if shape != (1, 5, 50, 50):
        raise ValueError(f"Unexpected score shape per frame: {shape}")
    score_values = int(np.prod(shape))
    expected_bytes = frame_count * score_values * np.dtype(np.float32).itemsize
    if args.scores.stat().st_size != expected_bytes:
        raise ValueError(
            f"Score file has {args.scores.stat().st_size} bytes; "
            f"expected {expected_bytes}"
        )

    RAW_PREDICTIONS = load_predictions(args.raw_predictions)
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    if RAW_PREDICTIONS.shape[0] < frame_count:
        raise ValueError(
            f"Only {RAW_PREDICTIONS.shape[0]} raw predictions for "
            f"{frame_count} score frames"
        )
    RAW_PREDICTIONS = RAW_PREDICTIONS[:frame_count]
    SCORES = np.memmap(
        args.scores,
        mode="r",
        dtype=np.float32,
        shape=(int(manifest["frames"]),) + shape,
    )[:frame_count, 0]

    targets = []
    masks = []
    segmentations = []
    for frame in range(frame_count):
        target, mask, segmentation = load_ground_truth(
            args.ground_truth, frame
        )
        targets.append(target)
        masks.append(mask)
        segmentations.append(segmentation)
    TARGETS = np.stack(targets)
    MASKS = np.stack(masks)
    SEGMENTATIONS = np.stack(segmentations)
    DX, BX, BEV_DIMENSION = build_grid(
        (-12.5, 12.5, 0.5), (-12.5, 12.5, 0.5)
    )
    EGO_FOOTPRINT = ego_footprint_pixels(DX, BX)

    worker_count = min(max(args.workers, 1), len(thresholds))
    if worker_count == 1:
        results = [evaluate_threshold(value) for value in thresholds]
    else:
        context = mp.get_context("fork")
        with context.Pool(worker_count) as pool:
            results = pool.map(evaluate_threshold, thresholds)
    results.sort(key=lambda item: item["threshold"])
    result = {
        "schema_version": 1,
        "method": "FP16 continuous occupancy score threshold sweep",
        "collision_optimizer": "deployment-matching independent-point BFGS",
        "frames": frame_count,
        "score_temporal_protocol": score_protocol,
        "raw_prediction_temporal_protocol": raw_protocol,
        "results": results,
        "best_avg_l2": min(results, key=lambda item: item["avg_l2_m"]),
        "best_avg_box_collision": min(
            results, key=lambda item: item["avg_box_collision_percent"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
