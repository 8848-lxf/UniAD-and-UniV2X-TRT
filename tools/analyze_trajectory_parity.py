#!/usr/bin/env python3

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def load_predictions(path):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            values = []
            for index in range(1, 7):
                values.append([float(row[f"x{index}"]), float(row[f"y{index}"])])
            rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--validation-info", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate = load_predictions(args.candidate)
    reference = load_predictions(args.reference)
    frame_count = min(len(candidate), len(reference))
    candidate = candidate[:frame_count]
    reference = reference[:frame_count]
    delta = candidate - reference
    point_l2 = np.linalg.norm(delta, axis=-1)
    frame_mean_l2 = point_l2.mean(axis=1)

    with open(args.validation_info, "rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    if len(infos) < frame_count:
        raise ValueError(f"Only {len(infos)} metadata frames for {frame_count} outputs")
    scene_first = np.zeros(frame_count, dtype=bool)
    previous_scene = None
    for index, info in enumerate(infos[:frame_count]):
        scene = info["scene_token"]
        scene_first[index] = scene != previous_scene
        previous_scene = scene

    worst = np.argsort(frame_mean_l2)[-20:][::-1]
    coordinate_mse = float(np.square(delta).mean())
    mean_squared_point_l2 = float(np.square(point_l2).mean())
    mean_point_l2 = float(point_l2.mean())
    result = {
        "schema_version": 1,
        "frames": frame_count,
        "scene_count": int(scene_first.sum()),
        "definitions": {
            "mean_point_l2_m": "mean(sqrt(dx^2 + dy^2)) over frames and six points",
            "coordinate_mse_m2": "mean(dx^2 and dy^2) over all coordinates",
            "mean_squared_point_l2_m2": "mean(dx^2 + dy^2) over frames and six points",
            "square_of_mean_point_l2_m2": "mean_point_l2_m squared; not an MSE identity",
        },
        "global": {
            "mean_point_l2_m": mean_point_l2,
            "coordinate_mse_m2": coordinate_mse,
            "mean_squared_point_l2_m2": mean_squared_point_l2,
            "square_of_mean_point_l2_m2": mean_point_l2 ** 2,
            "identity_check_mean_squared_point_l2_equals_2x_coordinate_mse": (
                mean_squared_point_l2 / (2.0 * coordinate_mse)
                if coordinate_mse else 1.0
            ),
            "frame_mean_point_l2_m": percentile_summary(frame_mean_l2),
            "by_horizon_mean_point_l2_m": point_l2.mean(axis=0).tolist(),
        },
        "scene_boundary": {
            "first_frames": int(scene_first.sum()),
            "first_frame_mean_point_l2_m": percentile_summary(frame_mean_l2[scene_first]),
            "non_first_frame_mean_point_l2_m": percentile_summary(frame_mean_l2[~scene_first]),
            "first_to_non_first_mean_ratio": float(
                frame_mean_l2[scene_first].mean() / frame_mean_l2[~scene_first].mean()
            ),
        },
        "worst_frames": [
            {
                "frame": int(index),
                "scene_token": infos[index]["scene_token"],
                "scene_first": bool(scene_first[index]),
                "frame_mean_point_l2_m": float(frame_mean_l2[index]),
                "frame_max_point_l2_m": float(point_l2[index].max()),
            }
            for index in worst
        ],
        "interpretation": (
            "Calibration cannot change FP32 parity. A scene-boundary ratio near one "
            "argues against scene-reset protocol mismatch; increasing error by horizon "
            "is consistent with accumulated numerical differences."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["global"], sort_keys=True))
    print(json.dumps(result["scene_boundary"], sort_keys=True))


if __name__ == "__main__":
    main()
