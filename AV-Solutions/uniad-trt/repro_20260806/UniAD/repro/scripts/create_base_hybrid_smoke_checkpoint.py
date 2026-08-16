#!/usr/bin/env python3

import argparse
import hashlib
import importlib
import json
import os
import random

import numpy as np
import torch
from mmcv import Config
from third_party.uniad_mmdet3d.models.builder import build_model


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("source_checkpoint")
    parser.add_argument("output_checkpoint")
    parser.add_argument("audit_json")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    cfg = Config.fromfile(args.config)
    if cfg.get("plugin", False):
        module_path = os.path.dirname(cfg.plugin_dir).replace("/", ".")
        importlib.import_module(module_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    target = model.state_dict()

    source_checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
    source = source_checkpoint.get("state_dict", source_checkpoint)
    transferred = []
    mismatched = []
    unexpected = []
    transferred_numel = 0
    target_numel = sum(value.numel() for value in target.values())

    for name, value in source.items():
        if name not in target:
            unexpected.append(name)
        elif tuple(value.shape) != tuple(target[name].shape):
            mismatched.append(
                {
                    "name": name,
                    "source_shape": list(value.shape),
                    "target_shape": list(target[name].shape),
                }
            )
        else:
            target[name] = value
            transferred.append(name)
            transferred_numel += value.numel()

    initialized = sorted(set(target) - set(transferred))
    output = {
        "meta": dict(source_checkpoint.get("meta", {})),
        "state_dict": target,
    }
    output["meta"]["hybrid_smoke_only"] = True
    output["meta"]["hybrid_source"] = os.path.abspath(args.source_checkpoint)
    output["meta"]["hybrid_warning"] = (
        "Engineering smoke checkpoint only; it is not a trained UniAD-tiny checkpoint."
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_checkpoint)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.audit_json)), exist_ok=True)
    torch.save(output, args.output_checkpoint)

    audit = {
        "purpose": "engineering_smoke_only",
        "valid_for_accuracy_comparison": False,
        "config": os.path.abspath(args.config),
        "source_checkpoint": os.path.abspath(args.source_checkpoint),
        "source_sha256": sha256(args.source_checkpoint),
        "output_checkpoint": os.path.abspath(args.output_checkpoint),
        "output_sha256": sha256(args.output_checkpoint),
        "source_keys": len(source),
        "target_keys": len(target),
        "transferred_keys": len(transferred),
        "transferred_numel": transferred_numel,
        "target_numel": target_numel,
        "transferred_numel_fraction": transferred_numel / target_numel,
        "initialized_keys": initialized,
        "shape_mismatches": mismatched,
        "unexpected_source_keys": sorted(unexpected),
    }
    with open(args.audit_json, "w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: audit[key] for key in (
        "source_keys", "target_keys", "transferred_keys",
        "transferred_numel_fraction", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
