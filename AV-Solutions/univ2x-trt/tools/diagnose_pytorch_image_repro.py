import argparse
import gc
import json
import os.path as osp
import sys

import torch


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
DEPLOY_ROOT = osp.join(REPO_ROOT, "deploy_int8", "UniV2X_deploy")
TRT_FUNCTIONS = osp.join(
    DEPLOY_ROOT, "projects", "mmdet3d_plugin", "uniad", "functions"
)
for source_root in (REPO_ROOT, DEPLOY_ROOT, TRT_FUNCTIONS):
    while source_root in sys.path:
        sys.path.remove(source_root)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TRT_FUNCTIONS)
sys.path.insert(0, DEPLOY_ROOT)

import projects.mmdet3d_plugin  # noqa: E402,F401

from cooperative_runtime import agent_image  # noqa: E402
from trt_runtime import build_trt_agent  # noqa: E402


def tensor_stats(value):
    value = value.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "l2": float(torch.linalg.vector_norm(value)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def run_once(config, checkpoint, frame, agent):
    _, model = build_trt_agent(config, checkpoint, agent)
    model = model.cuda().eval()
    agent_key = (
        "ego_agent_data" if agent == "ego" else "other_agent_data_dict"
    )
    agent_data = frame[agent_key]
    if agent == "infrastructure":
        agent_data = agent_data["model_other_agent_inf"]
    image = agent_image(agent_data, torch.device("cuda"))
    module_modes = {
        name: bool(module.training)
        for name, module in model.img_backbone.named_modules()
        if name in ("", "stem", "stage2", "stage3", "stage4", "stage5")
    }
    parameter_stats = {
        name: tensor_stats(parameter)
        for name, parameter in model.img_backbone.named_parameters()
        if name in (
            "stage4.OSA4_1.layers.0.conv.weight",
            "stage5.OSA5_1.layers.0.conv.weight",
        )
    }
    repeated = []
    with torch.no_grad():
        for _ in range(2):
            repeated.append([
                value.detach().cpu().clone()
                for value in model.extract_img_feat(image)
            ])
    repeat_error = []
    for expected, actual in zip(repeated[0], repeated[1]):
        difference = (expected.float() - actual.float()).abs()
        repeat_error.append({
            "mean_abs_error": float(difference.mean()),
            "max_abs_error": float(difference.max()),
        })
    result = {
        "module_training_modes": module_modes,
        "parameter_stats": parameter_stats,
        "image": tensor_stats(image),
        "outputs": [tensor_stats(value) for value in repeated[0]],
        "repeat_error": repeat_error,
    }
    del model, image, repeated
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("frame")
    parser.add_argument("--agent", choices=("ego", "infrastructure"), default="ego")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = torch.load(args.frame, map_location="cpu")
    runs = [
        run_once(args.config, args.checkpoint, frame, args.agent)
        for _ in range(2)
    ]
    build_error = []
    for expected, actual in zip(runs[0]["outputs"], runs[1]["outputs"]):
        build_error.append({
            "mean_delta": actual["mean"] - expected["mean"],
            "std_delta": actual["std"] - expected["std"],
            "l2_delta": actual["l2"] - expected["l2"],
        })
    report = {"runs": runs, "build_stat_delta": build_error}
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
