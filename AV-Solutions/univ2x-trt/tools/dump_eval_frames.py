import argparse
import copy
import os.path as osp
import sys

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mmcv
import torch
from mmcv.fileio.file_client import HardDiskBackend
from mmdet3d.datasets import build_dataset

import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from runtime_common import load_config


TENSOR_KEYS = (
    "img",
    "timestamp",
    "l2g_r_mat",
    "l2g_t",
    "command",
    "veh2inf_rt",
    "gt_lane_labels",
    "gt_lane_masks",
    "gt_segmentation",
    "gt_instance",
    "gt_occ_has_invalid_frame",
    "sdc_planning",
    "sdc_planning_mask",
)


def patch_disk_backend_file_objects():
    original_get = HardDiskBackend.get

    def compatible_get(backend, filepath):
        if hasattr(filepath, "read"):
            return filepath.read()
        return original_get(backend, filepath)

    HardDiskBackend.get = compatible_get


def unwrap(value):
    while isinstance(value, (list, tuple)):
        value = value[0]
    if hasattr(value, "data") and value.__class__.__name__ == "DataContainer":
        value = value.data
        while isinstance(value, (list, tuple)):
            value = value[0]
    return value


def serialize_agent(agent_data, include_ground_truth):
    meta = agent_data["img_metas"][0].data[0][0]
    serialized = {
        "img_metas": {
            "scene_token": str(meta["scene_token"]),
            "lidar2img": copy.deepcopy(meta["lidar2img"]),
            "can_bus": copy.deepcopy(meta["can_bus"]),
        }
    }
    keys = TENSOR_KEYS if include_ground_truth else TENSOR_KEYS[:5]
    for key in keys:
        if key not in agent_data:
            continue
        value = unwrap(agent_data[key])
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        serialized[key] = value.detach().cpu().clone()
    return serialized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.output_dir)
    patch_disk_backend_file_objects()
    cfg = load_config(args.config, workers=args.workers)
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    frame_limit = min(args.max_frames, len(dataset))
    progress = mmcv.ProgressBar(frame_limit)
    iterator = iter(loader)
    for frame_index in range(frame_limit):
        data = next(iterator)
        payload = {
            "ego_agent_data": serialize_agent(data["ego_agent_data"], True),
            "other_agent_data_dict": {
                "model_other_agent_inf": serialize_agent(
                    data["other_agent_data_dict"]["model_other_agent_inf"], True
                )
            },
        }
        torch.save(payload, osp.join(args.output_dir, "frame_%04d.pt" % frame_index))
        progress.update()


if __name__ == "__main__":
    main()
