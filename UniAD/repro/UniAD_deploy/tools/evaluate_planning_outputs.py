#!/usr/bin/env python3

"""Evaluate streamed TensorRT planning predictions with UniAD's metric formula."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from skimage.draw import polygon


N_FUTURE = 6
EGO_WIDTH = 1.85
EGO_HEIGHT = 4.084


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--x-bound", nargs=3, type=float, default=(-12.5, 12.5, 0.5))
    parser.add_argument("--y-bound", nargs=3, type=float, default=(-12.5, 12.5, 0.5))
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


def build_grid(x_bound, y_bound):
    bounds = np.asarray([x_bound, y_bound], dtype=np.float32)
    if np.any(bounds[:, 2] <= 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"Invalid BEV bounds: {bounds.tolist()}")
    dimensions_float = (bounds[:, 1] - bounds[:, 0]) / bounds[:, 2]
    dimensions = np.rint(dimensions_float).astype(np.int64)
    if not np.allclose(dimensions_float, dimensions):
        raise ValueError(f"BEV bounds do not form integral dimensions: {bounds.tolist()}")
    dx = bounds[:, 2]
    bx = bounds[:, 0] + dx / 2.0
    return dx, bx, dimensions


def ego_footprint_pixels(dx, bx):
    points = np.array(
        [
            [-EGO_HEIGHT / 2.0 + 0.5, EGO_WIDTH / 2.0],
            [EGO_HEIGHT / 2.0 + 0.5, EGO_WIDTH / 2.0],
            [EGO_HEIGHT / 2.0 + 0.5, -EGO_WIDTH / 2.0],
            [-EGO_HEIGHT / 2.0 + 0.5, -EGO_WIDTH / 2.0],
        ]
    )
    points = (points - bx) / dx
    points[:, [0, 1]] = points[:, [1, 0]]
    rows, columns = polygon(points[:, 1], points[:, 0])
    return np.stack([rows, columns], axis=-1)


def footprint_collision(trajectory, segmentation, dx, bev_dimension, ego_footprint):
    swapped = trajectory[:, [1, 0]] / dx
    footprint = swapped[:, None, :] + ego_footprint[None, :, :]
    rows = np.clip(footprint[:, :, 0].astype(np.int32), 0, bev_dimension[0] - 1)
    columns = np.clip(footprint[:, :, 1].astype(np.int32), 0, bev_dimension[1] - 1)
    result = np.zeros(N_FUTURE, dtype=np.bool_)
    for timestep in range(N_FUTURE):
        valid = (
            (rows[timestep] >= 0)
            & (rows[timestep] < bev_dimension[0])
            & (columns[timestep] >= 0)
            & (columns[timestep] < bev_dimension[1])
        )
        result[timestep] = np.any(
            segmentation[timestep, rows[timestep, valid], columns[timestep, valid]]
        )
    return result


def collision_metrics(prediction, target, segmentation, dx, bx, bev_dimension, ego_footprint):
    target_collision = footprint_collision(target, segmentation, dx, bev_dimension, ego_footprint)
    x = prediction[:, 0]
    y = prediction[:, 1]
    row = ((y - bx[0]) / dx[0]).astype(np.int64)
    column = ((x - bx[1]) / dx[1]).astype(np.int64)
    valid = (
        (row >= 0)
        & (row < bev_dimension[0])
        & (column >= 0)
        & (column < bev_dimension[1])
        & ~target_collision
    )
    point_collision = np.zeros(N_FUTURE, dtype=np.float64)
    timesteps = np.arange(N_FUTURE)[valid]
    point_collision[valid] = segmentation[timesteps, row[valid], column[valid]]
    box_collision = footprint_collision(
        prediction, segmentation, dx, bev_dimension, ego_footprint
    ).astype(np.float64)
    box_collision[target_collision] = 0.0
    return point_collision, box_collision


def load_ground_truth(root, frame):
    target = np.load(root / "sdc_planning" / f"{frame}.npy")[0, 0, :, :2]
    mask = np.load(root / "sdc_planning_mask" / f"{frame}.npy")[0, 0, :, :2]
    segmentation = np.load(root / "gt_segmentation" / f"{frame}.npy")[0, 1:7]
    return target.astype(np.float32), mask.astype(np.float32), segmentation.astype(np.bool_)


def main():
    args = parse_args()
    dx, bx, bev_dimension = build_grid(args.x_bound, args.y_bound)
    ego_footprint = ego_footprint_pixels(dx, bx)
    predictions = load_predictions(args.predictions)
    frame_count = predictions.shape[0]
    l2_sum = np.zeros(N_FUTURE, dtype=np.float64)
    point_collision_sum = np.zeros(N_FUTURE, dtype=np.float64)
    box_collision_sum = np.zeros(N_FUTURE, dtype=np.float64)

    for frame, prediction in enumerate(predictions):
        target, mask, segmentation = load_ground_truth(args.ground_truth, frame)
        if tuple(segmentation.shape[-2:]) != tuple(bev_dimension):
            raise ValueError(
                f"Frame {frame} segmentation shape {segmentation.shape[-2:]} does not match "
                f"configured BEV dimensions {tuple(bev_dimension)}"
            )
        l2_sum += np.sqrt(np.sum((prediction - target) ** 2 * mask, axis=-1))
        point_collision, box_collision = collision_metrics(
            prediction, target, segmentation, dx, bx, bev_dimension, ego_footprint
        )
        point_collision_sum += point_collision
        box_collision_sum += box_collision

    l2 = l2_sum / frame_count
    point_collision_percent = point_collision_sum / frame_count * 100.0
    box_collision_percent = box_collision_sum / frame_count * 100.0
    result = {
        "schema_version": 1,
        "frames": frame_count,
        "metric_source": "UniAD PlanningMetric",
        "x_bound": list(args.x_bound),
        "y_bound": list(args.y_bound),
        "bev_dimension": bev_dimension.tolist(),
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
        mean_point_l2 = float(np.linalg.norm(delta, axis=-1).mean())
        coordinate_mse = float(np.square(delta).mean())
        result["planning_reference_frames"] = frame_count
        result["planning_output_mean_point_l2_m"] = mean_point_l2
        result["planning_output_coordinate_mse_m2"] = coordinate_mse
        result["planning_output_mean_squared_point_l2_m2"] = 2.0 * coordinate_mse
        # Backward-compatible keys for reports generated before the units were
        # made explicit. NVIDIA's table calls this value "planning MSE", but
        # its documentation defines it as average trajectory-point L2.
        result["planning_reference_avg_l2_m"] = mean_point_l2
        result["planning_reference_coordinate_mse"] = coordinate_mse
        result["planning_mse"] = mean_point_l2
        result["planning_mse_definition"] = (
            "NVIDIA table semantics: mean Euclidean L2 distance in meters "
            "between TensorRT and PyTorch trajectory points; not a squared error"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
