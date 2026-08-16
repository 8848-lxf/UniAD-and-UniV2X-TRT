import argparse
import json
import os

import torch
import projects.mmdet3d_plugin  # noqa: F401

from trt_runtime import (
    INFRASTRUCTURE_OUTPUT_NAMES,
    EGO_INPUT_NAMES,
    OUTPUT_NAMES,
    build_trt_agent,
    load_export_inputs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("input_npz")
    parser.add_argument("--agent", choices=("ego", "infrastructure"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coop-input-npz")
    args = parser.parse_args()

    _, model = build_trt_agent(args.config, args.checkpoint, args.agent)
    model = model.cuda().eval()
    inputs = load_export_inputs(
        args.input_npz, torch.device("cuda"), args.coop_input_npz
    )
    torch.cuda.synchronize()
    with torch.no_grad():
        outputs = model.forward_uniad_trt(*inputs)
    torch.cuda.synchronize()

    output_names = (
        OUTPUT_NAMES if args.agent == "ego" else INFRASTRUCTURE_OUTPUT_NAMES
    )
    if len(outputs) != len(output_names):
        raise RuntimeError((len(outputs), len(output_names)))
    report = {
        "agent": args.agent,
        "gpu": torch.cuda.get_device_name(0),
        "max_memory_mib": torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
        "outputs": {},
    }
    for name, value in zip(output_names, outputs):
        finite = torch.isfinite(value) if value.is_floating_point() else None
        report["outputs"][name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "all_finite": bool(finite.all().item()) if finite is not None else True,
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
