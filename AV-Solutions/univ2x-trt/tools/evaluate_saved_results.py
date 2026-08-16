import argparse
import io
import json
import os
import os.path as osp
import sys

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mmcv
import numpy as np
import torch
from mmcv.fileio.file_client import HardDiskBackend
from mmdet3d.datasets import build_dataset

from runtime_common import load_config


VEHICLE_LABELS = (0, 1, 2, 3, 4, 6, 7)
TRAJECTORY_KEYS = (
    ("traj_0", "traj_scores_0"),
    ("traj_1", "traj_scores_1"),
    ("traj", "traj_scores"),
)


def patch_disk_backend_file_objects():
    original_get = HardDiskBackend.get

    def compatible_get(backend, filepath):
        if hasattr(filepath, "read"):
            return filepath.read()
        return original_get(backend, filepath)

    HardDiskBackend.get = compatible_get


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


def load_pickle_on_cpu(path):
    original = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda value: torch.load(
        io.BytesIO(value), map_location="cpu"
    )
    try:
        return mmcv.load(path)
    finally:
        torch.storage._load_from_bytes = original


def repair_trajectory_alignment(result):
    labels = result["labels_3d"].long()
    box_count = labels.numel()
    vehicle_mask = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in VEHICLE_LABELS:
        vehicle_mask |= labels == class_id
    vehicle_indices = torch.where(vehicle_mask)[0]
    vehicle_count = vehicle_indices.numel()
    repaired = False

    for trajectory_key, score_key in TRAJECTORY_KEYS:
        trajectories = result[trajectory_key]
        scores = result[score_key]
        if trajectories.shape[0] == box_count + 1:
            continue
        if trajectories.shape[0] != vehicle_count + 1:
            raise RuntimeError({
                "token": result.get("token"),
                "boxes": box_count,
                "vehicles": vehicle_count,
                "trajectories": trajectories.shape[0],
            })
        aligned_trajectories = trajectories.new_zeros(
            (box_count + 1, *trajectories.shape[1:])
        )
        aligned_scores = scores.new_zeros((box_count + 1, *scores.shape[1:]))
        if vehicle_count:
            aligned_trajectories[vehicle_indices] = trajectories[:vehicle_count]
            aligned_scores[vehicle_indices] = scores[:vehicle_count]
        aligned_trajectories[-1] = trajectories[-1]
        aligned_scores[-1] = scores[-1]
        result[trajectory_key] = aligned_trajectories
        result[score_key] = aligned_scores
        repaired = True
    return repaired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    mmcv.mkdir_or_exist(args.output_dir)
    cfg = load_config(args.config, workers=args.workers)
    patch_disk_backend_file_objects()
    dataset = build_dataset(cfg.data.test)
    bundle = load_pickle_on_cpu(args.results)
    if len(bundle["bbox_results"]) != len(dataset):
        raise RuntimeError((len(bundle["bbox_results"]), len(dataset)))

    repaired_frames = sum(
        int(repair_trajectory_alignment(result))
        for result in bundle["bbox_results"]
    )
    repaired_path = osp.join(args.output_dir, "results_repaired.pkl")
    mmcv.dump(bundle, repaired_path)

    eval_dir = osp.join(args.output_dir, "full_task_eval")
    mmcv.mkdir_or_exist(eval_dir)
    detail = dataset.evaluate(
        bundle,
        metric=["bbox"],
        jsonfile_prefix=eval_dir,
        pipeline=cfg.get("evaluation", {}).get("pipeline"),
    )
    report = {
        "source_results": os.path.abspath(args.results),
        "repaired_results": os.path.abspath(repaired_path),
        "dataset_frames": len(dataset),
        "repaired_frames": repaired_frames,
        "metrics": scalarize(detail),
    }
    with open(osp.join(args.output_dir, "dataset_metrics.json"), "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=True)
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
