#!/usr/bin/env python3
"""Extract and verify the full-Python optimized trajectory reference."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=6018)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main():
    args = parse_args()
    with args.results.open("rb") as handle:
        payload = pickle.load(handle)
    results = payload["bbox_results"]
    if len(results) < args.frames:
        raise ValueError(
            "results contain %d frames, need %d" % (len(results), args.frames)
        )

    trajectories = []
    maximum_gt_delta = 0.0
    mismatched_ground_truth_frames = []
    for frame, result in enumerate(results[:args.frames]):
        trajectory = as_numpy(result["planning_traj"])[0, :6, :2]
        result_ground_truth = as_numpy(result["planning_traj_gt"][0])
        deployment_ground_truth = np.load(
            args.ground_truth / "sdc_planning" / ("%d.npy" % frame)
        )
        delta = float(np.max(np.abs(
            result_ground_truth[..., :6, :2]
            - deployment_ground_truth[..., :6, :2]
        )))
        maximum_gt_delta = max(maximum_gt_delta, delta)
        if delta > 1.0e-6:
            mismatched_ground_truth_frames.append(frame)
        if not np.isfinite(trajectory).all():
            raise ValueError("non-finite optimized trajectory at frame %d" % frame)
        trajectories.append(trajectory.astype(np.float32))

    if mismatched_ground_truth_frames:
        raise RuntimeError(
            "full-Python and deployment sequences do not align; first mismatches: %r"
            % mismatched_ground_truth_frames[:10]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame"] + [
        axis + str(step)
        for step in range(1, 7)
        for axis in ("x", "y")
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame, trajectory in enumerate(trajectories):
            row = {"frame": frame}
            for step, point in enumerate(trajectory, start=1):
                row["x%d" % step] = "%.9g" % float(point[0])
                row["y%d" % step] = "%.9g" % float(point[1])
            writer.writerow(row)

    report = {
        "schema_version": 1,
        "source": str(args.results.resolve()),
        "frames_available": len(results),
        "frames_extracted": args.frames,
        "trajectory_protocol": "full Python simple_test with use_col_optim=True",
        "ground_truth_alignment_max_abs_delta": maximum_gt_delta,
        "ground_truth_mismatched_frames": 0,
        "output": str(args.output.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
