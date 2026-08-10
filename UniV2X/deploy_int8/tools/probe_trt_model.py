import argparse
import json
import os

import torch
from mmcv import Config
import projects.mmdet3d_plugin  # noqa: F401
from third_party.uniad_mmdet3d.models.builder import build_model

from trt_config import make_trt_agent_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--agent", choices=("ego", "infrastructure"), required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.agent == "ego":
        source = cfg.model_ego_agent
        prefix = "model_ego_agent."
        cooperative = True
    else:
        source = cfg.model_other_agent_inf
        prefix = "model_other_agent_inf."
        cooperative = False

    agent_cfg = make_trt_agent_config(source, cooperative=cooperative)
    # This key is consumed by the UniV2X wrapper added after base compatibility.
    agent_cfg.pop("univ2x_cooperative_graph")
    agent_cfg["train_cfg"] = None
    model = build_model(agent_cfg, test_cfg=cfg.get("test_cfg"))

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    incompatible = model.load_state_dict(state, strict=False)
    model_keys = set(model.state_dict())
    state_keys = set(state)
    report = {
        "agent": args.agent,
        "checkpoint": os.path.realpath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("meta", {}).get("epoch"),
        "source_tensor_count": len(state),
        "model_tensor_count": len(model_keys),
        "matched_tensor_count": len(model_keys & state_keys),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "matched_parameter_elements": int(sum(
            value.numel() for key, value in model.state_dict().items()
            if key in state and tuple(value.shape) == tuple(state[key].shape)
        )),
        "model_parameter_elements": int(sum(
            value.numel() for value in model.state_dict().values()
        )),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)
    summary = dict(report)
    summary["missing_keys"] = report["missing_keys"][:20]
    summary["unexpected_keys"] = report["unexpected_keys"][:20]
    summary["missing_key_count"] = len(report["missing_keys"])
    summary["unexpected_key_count"] = len(report["unexpected_keys"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
