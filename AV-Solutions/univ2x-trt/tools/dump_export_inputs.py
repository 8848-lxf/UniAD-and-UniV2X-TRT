import argparse
import json
import os

import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from runtime_common import import_plugin


def tensor_from_list(agent_data, key):
    value = agent_data[key]
    if isinstance(value, list):
        value = value[0]
    return value.detach().cpu().numpy()


def dump_agent(agent_data, output_path):
    meta = agent_data["img_metas"][0].data[0][0]
    image = agent_data["img"][0].data[0].detach().cpu().numpy().astype(np.float32)
    lidar2img = np.asarray(meta["lidar2img"], dtype=np.float32)[None]
    # The image feature is broadcast to six learned camera slots, while the
    # source graph keeps only the first projection slot visible.
    if lidar2img.shape[1] == 1:
        padded_lidar2img = np.zeros((1, 6, 4, 4), dtype=np.float32)
        padded_lidar2img[:, 0] = lidar2img[:, 0]
        lidar2img = padded_lidar2img

    can_bus = np.asarray(meta["can_bus"], dtype=np.float32).copy()
    can_bus[:3] = 0.0
    can_bus[-1] = 0.0
    scene_bytes = np.zeros(32, dtype=np.uint8)
    encoded = str(meta["scene_token"]).encode("utf-8")[:32]
    scene_bytes[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)

    values = {
        "img": image,
        "img_metas_can_bus": can_bus,
        "img_metas_lidar2img": lidar2img,
        "image_shape": np.asarray(image.shape[-2:], dtype=np.float32),
        "img_metas_scene_token": scene_bytes,
        "timestamp": tensor_from_list(agent_data, "timestamp").astype(np.float32),
        "l2g_r_mat": tensor_from_list(agent_data, "l2g_r_mat").astype(np.float32),
        "l2g_t": tensor_from_list(agent_data, "l2g_t").astype(np.float32),
        "gt_lane_labels": tensor_from_list(agent_data, "gt_lane_labels").astype(np.int64),
        "gt_lane_masks": tensor_from_list(agent_data, "gt_lane_masks").astype(np.uint8),
        "gt_segmentation": tensor_from_list(agent_data, "gt_segmentation").astype(np.int64),
        "gt_instance": tensor_from_list(agent_data, "gt_instance").astype(np.int64),
        "command": tensor_from_list(agent_data, "command").astype(np.int64),
        "veh2inf_rt": tensor_from_list(agent_data, "veh2inf_rt").astype(np.float32),
    }
    np.savez(output_path, **values)
    return {
        "sample_idx": meta["sample_idx"],
        "scene_token": meta["scene_token"],
        "tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in values.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    batch = next(iter(loader))
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "ego": dump_agent(
            batch["ego_agent_data"], os.path.join(args.output_dir, "ego.npz")
        ),
        "infrastructure": dump_agent(
            batch["other_agent_data_dict"]["model_other_agent_inf"],
            os.path.join(args.output_dir, "infrastructure.npz"),
        ),
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
