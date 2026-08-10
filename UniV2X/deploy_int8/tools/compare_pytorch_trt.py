import argparse
import gc
import json
import os

import torch

import projects.mmdet3d_plugin  # noqa: F401

from trt_engine import TensorRTEngine
from trt_runtime import (
    EGO_INPUT_NAMES,
    INFRASTRUCTURE_OUTPUT_NAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    build_trt_agent,
    load_export_inputs,
)


def compare_tensor(reference, actual):
    result = {
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
    }
    if tuple(reference.shape) != tuple(actual.shape):
        result["shape_match"] = False
        return result
    result["shape_match"] = True
    reference = reference.cpu()
    actual = actual.cpu()
    if reference.is_floating_point():
        difference = (reference.float() - actual.float()).abs()
        result.update({
            "reference_min": float(reference.min()),
            "reference_max": float(reference.max()),
            "actual_min": float(actual.min()),
            "actual_max": float(actual.max()),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "allclose_rtol_1e-3_atol_1e-4": bool(torch.allclose(
                reference.float(), actual.float(), rtol=1.0e-3, atol=1.0e-4
            )),
        })
    else:
        result["mismatch_count"] = int((reference != actual).sum())
        result["element_count"] = reference.numel()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("engine")
    parser.add_argument("plugin")
    parser.add_argument("input_npz")
    parser.add_argument("--agent", choices=("ego", "infrastructure"), required=True)
    parser.add_argument("--coop-input-npz")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_names = EGO_INPUT_NAMES if args.agent == "ego" else INPUT_NAMES
    output_names = OUTPUT_NAMES if args.agent == "ego" else INFRASTRUCTURE_OUTPUT_NAMES

    _, model = build_trt_agent(args.config, args.checkpoint, args.agent)
    model = model.cuda().eval()
    inputs = load_export_inputs(
        args.input_npz, torch.device("cuda"), args.coop_input_npz
    )
    with torch.no_grad():
        reference_values = model.forward_uniad_trt(*inputs)
    reference = {
        name: value.detach().cpu()
        for name, value in zip(output_names, reference_values)
    }
    del reference_values, model, inputs
    gc.collect()
    torch.cuda.empty_cache()

    engine = TensorRTEngine(args.engine, args.plugin)
    engine_inputs = load_export_inputs(
        args.input_npz, torch.device("cuda"), args.coop_input_npz
    )
    actual = engine.infer(dict(zip(input_names, engine_inputs)))
    common_outputs = [name for name in engine.output_names if name in reference]
    extra_engine_outputs = [
        name for name in engine.output_names if name not in reference
    ]
    missing_engine_outputs = [
        name for name in output_names if name not in actual
    ]
    comparisons = {
        name: compare_tensor(reference[name], actual[name])
        for name in common_outputs
    }
    result = {
        "agent": args.agent,
        "engine": os.path.abspath(args.engine),
        "gpu": torch.cuda.get_device_name(0),
        "extra_engine_outputs": extra_engine_outputs,
        "missing_engine_outputs": missing_engine_outputs,
        "outputs": comparisons,
        "shape_mismatches": [
            name for name, value in comparisons.items()
            if not value["shape_match"]
        ],
        "integer_mismatches": {
            name: value["mismatch_count"]
            for name, value in comparisons.items()
            if value.get("mismatch_count", 0)
        },
        "floating_not_allclose": [
            name for name, value in comparisons.items()
            if value.get("allclose_rtol_1e-3_atol_1e-4") is False
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        "shape_mismatches": result["shape_mismatches"],
        "integer_mismatches": result["integer_mismatches"],
        "floating_not_allclose": result["floating_not_allclose"],
        "output": os.path.abspath(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
