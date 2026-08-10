import argparse
import json
import os

import numpy as np
import torch
from mmcv import Config

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from runtime_common import import_plugin
from mmdet3d.datasets import build_dataset


def describe(value):
    if hasattr(value, "data") and value.__class__.__name__ == "DataContainer":
        return {
            "kind": "DataContainer",
            "cpu_only": value.cpu_only,
            "stack": value.stack,
            "data": describe(value.data),
        }
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {
            "kind": value.__class__.__name__,
            "length": len(value),
            "items": [describe(item) for item in value[:2]],
        }
    if isinstance(value, torch.Tensor):
        return {
            "kind": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    return {"kind": value.__class__.__name__, "repr": repr(value)[:300]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
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
    report = {"dataset_length": len(dataset), "batch": describe(next(iter(loader)))}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
