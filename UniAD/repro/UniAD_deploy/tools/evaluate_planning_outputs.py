#!/usr/bin/env python3

"""Evaluate streamed TensorRT planning predictions with UniAD's metric formula."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from skimage.draw import polygon


N_FUTURE = 6
DX = np.array([0.5, 0.5], dtype=np.float32)
BX = np.array([-12.25, -12.25], dtype=np.float32)
BEV_DIMENSION = np.array([50, 50], dtype=np.int64)
EGO_WIDTH = 1.85
EGO_HEIGHT = 4.084


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-predictions", type=Path)
    return parser.parse_args()


def load_predictions(path):
    rows = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [axis + str(step) for step in range(1, N_FUTURE + 1) for axis in ("x", "y")]
        if reader.fieldnames != ["frame"] + expected:
            raise ValueError(f"Unexpected prediction columns in {path}: {reader.fieldnames}")
        for row in reader:
            frame = int(row["frame"])
            values = np.asarray([float(row[key]) for key in expected], dtype=np.float32)
            rows[frame] = values.reshape(N_FUTURE, 2)
    if not rows:
        raise ValueError(f"No predictions found in {path}")
    expected_frames = list(range(len(rows)))
    if sorted(rows) != expected_frames:
        raise ValueError("Prediction frame ids must be contiguous and start at zero")
    return np.stack([rows[frame] for frame in expected_frames])


def ego_footprint_pixels():
    points = np.array(
        [
            [-EGO_HEIGHT / 2.0 + 0.5, EGO_WIDTH / 2.0],
            [EGO_HEIGHT / 2.0 + 0.5, EGO_WIDTH / 2.0],
            [EGO_HEIGHT / 2.0 + 0.5, -EGO_WIDTH / 2.0],
            [-EGO_HEIGHT / 2.0 + 0.5, -EGO_WIDTH / 2.0],
        ]
    )
    points = (points - BX) / DX
    points[:, [0, 1]] = points[:, [1, 0]]
    rows, columns = polygon(points[:, 1], points[:, 0])
    return np.stack([rows, columns], axis=-1)


EGO_FOOTPRINT = ego_footprint_pixels()


def footprint_collision(trajectory, segmentation):
    swapped = trajectory[:, [1, 0]] / DX
    footprint = swapped[:, None, :] + EGO_FOOTPRINT[None, :, :]
    rows = np.clip(footprint[:, :, 0].astype(np.int32), 0, BEV_DIMENSION[0] - 1)
    columns = np.clip(footprint[:, :, 1].astype(np.int32), 0, BEV_DIMENSION[1] - 1)
    result = np.zeros(N_FUTURE, dtype=np.bool_)
    for timestep in range(N_FUTURE):
        valid = (
            (rows[timestep] >= 0)
            & (rows[timestep] < BEV_DIMENSION[0])
            & (columns[timestep] >= 0)
            & (columns[timestep] < BEV_DIMENSION[1])
        )
        result[timestep] = np.any(
            segmentation[timestep, rows[timestep, valid], columns[timestep, valid]]
        )
    return result


def collision_metrics(prediction, target, segmentation):
    target_collision = footprint_collision(target, segmentation)
    x = prediction[:, 0]
    y = prediction[:, 1]
    row = ((y - BX[0]) / DX[0]).astype(np.int64)
    column = ((x - BX[1]) / DX[1]).astype(np.int64)
    valid = (
        (row >= 0)
        & (row < BEV_DIMENSION[0])
        & (column >= 0)
        & (column < BEV_DIMENSION[1])
        & ~target_collision
    )
    point_collision = np.zeros(N_FUTURE, dtype=np.float64)
    timesteps = np.arange(N_FUTURE)[valid]
    point_collision[valid] = segmentation[timesteps, row[valid], column[valid]]
    box_collision = footprint_collision(prediction, segmentation).astype(np.float64)
    box_collision[target_collision] = 0.0
    return point_collision, box_collision


def load_ground_truth(root, frame):
    target = np.load(root / "sdc_planning" / f"{frame}.npy")[0, 0, :, :2]
    mask = np.load(root / "sdc_planning_mask" / f"{frame}.npy")[0, 0, :, :2]
    segmentation = np.load(root / "gt_segmentation" / f"{frame}.npy")[0, 1:7]
    return target.astype(np.float32), mask.astype(np.float32), segmentation.astype(np.bool_)


def main():
    args = parse_args()
    predictions = load_predictions(args.predictions)
    frame_count = predictions.shape[0]
    l2_sum = np.zeros(N_FUTURE, dtype=np.float64)
    point_collision_sum = np.zeros(N_FUTURE, dtype=np.float64)
    box_collision_sum = np.zeros(N_FUTURE, dtype=np.float64)

    for frame, prediction in enumerate(predictions):
        target, mask, segmentation = load_ground_truth(args.ground_truth, frame)
        l2_sum += np.sqrt(np.sum((prediction - target) ** 2 * mask, axis=-1))
        point_collision, box_collision = collision_metrics(prediction, target, segmentation)
        point_collision_sum += point_collision
        box_collision_sum += box_collision

    l2 = l2_sum / frame_count
    point_collision_percent = point_collision_sum / frame_count * 100.0
    box_collision_percent = box_collision_sum / frame_count * 100.0
    result = {
        "schema_version": 1,
        "frames": frame_count,
        "metric_source": "UniAD PlanningMetric with tiny bounds [-12.5, 12.5, 0.5]",
        "l2_by_timestep_m": l2.tolist(),
        "avg_l2_m": float(np.mean(l2)),
        "point_collision_by_timestep_percent": point_collision_percent.tolist(),
        "avg_point_collision_percent": float(np.mean(point_collision_percent)),
        "box_collision_by_timestep_percent": box_collision_percent.tolist(),
        "avg_box_collision_percent": float(np.mean(box_collision_percent)),
    }

    if args.reference_predictions:
        reference = load_predictions(args.reference_predictions)
        if reference.shape[0] < predictions.shape[0] or reference.shape[1:] != predictions.shape[1:]:
            raise ValueError(
                f"Reference shape {reference.shape} cannot cover prediction shape {predictions.shape}"
            )
        reference = reference[:frame_count]
        delta = predictions - reference
        result["planning_reference_frames"] = frame_count
        result["planning_reference_avg_l2_m"] = float(
            np.linalg.norm(delta, axis=-1).mean()
        )
        result["planning_reference_coordinate_mse"] = float(np.square(delta).mean())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
