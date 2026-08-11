import argparse
import gc
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEPLOY_ROOT = os.path.join(REPO_ROOT, "deploy_int8", "UniV2X_deploy")
TRT_FUNCTIONS = os.path.join(
    DEPLOY_ROOT, "projects", "mmdet3d_plugin", "uniad", "functions"
)
for source_root in (REPO_ROOT, DEPLOY_ROOT, TRT_FUNCTIONS):
    while source_root in sys.path:
        sys.path.remove(source_root)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TRT_FUNCTIONS)
sys.path.insert(0, DEPLOY_ROOT)

import torch

import projects.mmdet3d_plugin  # noqa: F401

from trt_engine import TensorRTEngine
from trt_runtime import (
    COOP_TRACK_INPUT_NAMES,
    EGO_INPUT_NAMES,
    INFRASTRUCTURE_OUTPUT_NAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    TRACK_OUTPUT_NAMES,
    apply_dynamic_map_postprocess,
    build_trt_agent,
    load_export_inputs,
)


EXPECTED_RUNTIME_DROPPED_OUTPUTS = {
    "drivable_intersection", "drivable_union",
    "lanes_intersection", "lanes_union",
    "divider_intersection", "divider_union",
    "crossing_intersection", "crossing_union",
    "contour_intersection", "contour_union",
}


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
        result["reference_nonzero"] = int(torch.count_nonzero(reference))
        result["actual_nonzero"] = int(torch.count_nonzero(actual))
    return result


def pad_fixed_inputs(values, input_names, fixed_track_count, fixed_coop_count):
    inputs = dict(zip(input_names, values))

    if fixed_track_count:
        current = int(inputs["prev_track_intances0"].shape[0])
        if current > fixed_track_count:
            raise ValueError(
                f"Track count {current} exceeds fixed capacity {fixed_track_count}"
            )
        for output_name in TRACK_OUTPUT_NAMES:
            name = output_name[:-4]
            value = inputs[name]
            padding = fixed_track_count - current
            if padding:
                inputs[name] = torch.cat([
                    value,
                    torch.full(
                        (padding,) + value.shape[1:],
                        -10000,
                        dtype=value.dtype,
                        device=value.device,
                    ),
                ])

    if fixed_coop_count:
        current = int(inputs["coop_track_intances0"].shape[0])
        if current > fixed_coop_count:
            raise ValueError(
                f"Cooperative track count {current} exceeds fixed capacity "
                f"{fixed_coop_count}"
            )
        padding = fixed_coop_count - current
        if padding:
            padded_names = [
                name for name in COOP_TRACK_INPUT_NAMES
                if name.startswith("coop_track_intances")
            ] + ["coop_match_vehicle_index"]
            for name in padded_names:
                value = inputs[name]
                fill_value = 2147483647 if name == "coop_match_vehicle_index" else 0
                inputs[name] = torch.cat([
                    value,
                    torch.full(
                        (padding,) + value.shape[1:],
                        fill_value,
                        dtype=value.dtype,
                        device=value.device,
                    ),
                ])

    return tuple(inputs[name] for name in input_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("engine")
    parser.add_argument("plugin")
    parser.add_argument("input_npz")
    parser.add_argument("--agent", choices=("ego", "infrastructure"), required=True)
    parser.add_argument("--coop-input-npz")
    parser.add_argument("--fixed-track-count", type=int, default=0)
    parser.add_argument("--fixed-coop-count", type=int, default=0)
    parser.add_argument(
        "--agent-normalization-epsilon", type=float, default=0.0009765625
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_names = EGO_INPUT_NAMES if args.agent == "ego" else INPUT_NAMES
    output_names = OUTPUT_NAMES if args.agent == "ego" else INFRASTRUCTURE_OUTPUT_NAMES

    _, model = build_trt_agent(args.config, args.checkpoint, args.agent)
    if args.agent == "ego":
        model.cross_agent_query_interaction.normalization_epsilon = (
            args.agent_normalization_epsilon
        )
    model = model.cuda().eval()
    inputs = load_export_inputs(
        args.input_npz, torch.device("cuda"), args.coop_input_npz
    )
    inputs = pad_fixed_inputs(
        inputs,
        input_names,
        args.fixed_track_count,
        args.fixed_coop_count,
    )
    with torch.no_grad():
        reference_values = model.forward_uniad_trt(*inputs)
    reference = {
        name: value.detach().cpu()
        for name, value in zip(output_names, reference_values)
    }
    apply_dynamic_map_postprocess(reference)
    del reference_values, model, inputs
    gc.collect()
    torch.cuda.empty_cache()

    engine = TensorRTEngine(args.engine, args.plugin)
    engine_inputs = load_export_inputs(
        args.input_npz, torch.device("cuda"), args.coop_input_npz
    )
    engine_inputs = pad_fixed_inputs(
        engine_inputs,
        input_names,
        args.fixed_track_count,
        args.fixed_coop_count,
    )
    actual = engine.infer(dict(zip(input_names, engine_inputs)))
    apply_dynamic_map_postprocess(actual)
    common_outputs = [name for name in engine.output_names if name in reference]
    if (
        "lane_pred" in actual
        and "lane_pred" in reference
        and "lane_pred" not in common_outputs
    ):
        common_outputs.append("lane_pred")
    extra_engine_outputs = [
        name for name in engine.output_names if name not in reference
    ]
    expected_dropped_outputs = sorted(
        name for name in EXPECTED_RUNTIME_DROPPED_OUTPUTS if name not in actual
    )
    missing_engine_outputs = [
        name for name in output_names
        if name not in actual and name not in EXPECTED_RUNTIME_DROPPED_OUTPUTS
    ]
    comparisons = {
        name: compare_tensor(reference[name], actual[name])
        for name in common_outputs
    }
    result = {
        "agent": args.agent,
        "agent_normalization_epsilon": (
            args.agent_normalization_epsilon if args.agent == "ego" else None
        ),
        "engine": os.path.abspath(args.engine),
        "gpu": torch.cuda.get_device_name(0),
        "extra_engine_outputs": extra_engine_outputs,
        "expected_dropped_outputs": expected_dropped_outputs,
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
