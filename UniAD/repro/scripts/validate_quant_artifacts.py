#!/usr/bin/env python3

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


EXPECTED_INPUTS = [
    "prev_track_intances0",
    "prev_track_intances1",
    "prev_track_intances3",
    "prev_track_intances4",
    "prev_track_intances5",
    "prev_track_intances6",
    "prev_track_intances8",
    "prev_track_intances9",
    "prev_track_intances11",
    "prev_track_intances12",
    "prev_track_intances13",
    "prev_timestamp",
    "prev_l2g_r_mat",
    "prev_l2g_t",
    "prev_bev",
    "timestamp",
    "l2g_r_mat",
    "l2g_t",
    "img",
    "img_metas_can_bus",
    "img_metas_lidar2img",
    "command",
    "use_prev_bev",
    "max_obj_id",
]

EXPECTED_TRT_PLUGINS = {
    "InverseTRT",
    "ModulatedDeformableConv2dTRT",
    "MultiScaleDeformableAttnTRT",
    "RotateTRT",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-onnx", required=True, type=Path)
    parser.add_argument("--quant-onnx", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--skip-onnx-checker",
        action="store_true",
        help="record checker as skipped; use only when TensorRT parser/build is the structural gate",
    )
    return parser.parse_args()


def graph_inputs(model):
    initializers = {item.name for item in model.graph.initializer}
    return [item.name for item in model.graph.input if item.name not in initializers]


def check_model_with_expected_plugins(model):
    plugin_nodes = []
    original_domains = []
    for node in model.graph.node:
        if node.op_type in EXPECTED_TRT_PLUGINS:
            original_domains.append((node, node.domain))
            node.domain = "nvidia.trt.plugin"
            plugin_nodes.append(node.op_type)
    added_opset = False
    if plugin_nodes and not any(
            item.domain == "nvidia.trt.plugin" for item in model.opset_import):
        model.opset_import.add(domain="nvidia.trt.plugin", version=1)
        added_opset = True
    try:
        onnx.checker.check_model(model, check_custom_domain=False)
    finally:
        for node, domain in original_domains:
            node.domain = domain
        if added_opset:
            del model.opset_import[-1]
    return sorted(set(plugin_nodes))


def summarize_model(path, run_checker=True):
    model = onnx.load(str(path), load_external_data=False)
    plugin_nodes = (
        check_model_with_expected_plugins(model)
        if run_checker
        else sorted({
            node.op_type for node in model.graph.node
            if node.op_type in EXPECTED_TRT_PLUGINS
        })
    )
    operators = collections.Counter(node.op_type for node in model.graph.node)
    domains = collections.Counter(node.domain or "ai.onnx" for node in model.graph.node)
    initializers_by_type = collections.Counter(
        TensorProto.DataType.Name(initializer.data_type)
        for initializer in model.graph.initializer
    )
    scales = []
    initializer_map = {initializer.name: initializer for initializer in model.graph.initializer}
    producer_map = {
        output: node for node in model.graph.node for output in node.output
    }
    matmul_int8_weight_inputs = []
    matmul_dq_activation_inputs = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        for input_name in node.input:
            producer = producer_map.get(input_name)
            if producer is None or producer.op_type != "DequantizeLinear":
                continue
            quantized_source = initializer_map.get(producer.input[0])
            if (quantized_source is not None
                    and quantized_source.data_type == TensorProto.INT8):
                matmul_int8_weight_inputs.append(node.name)
            else:
                matmul_dq_activation_inputs.append(node.name)
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear") or len(node.input) < 2:
            continue
        initializer = initializer_map.get(node.input[1])
        if initializer is not None:
            if initializer.data_location == TensorProto.EXTERNAL:
                raise AssertionError(
                    f"Quantization scale is external and was not loaded: {initializer.name}"
                )
            scales.append(numpy_helper.to_array(initializer).astype(np.float64).reshape(-1))
    scale_values = np.concatenate(scales) if scales else np.asarray([], dtype=np.float64)
    return {
        "path": str(path),
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": graph_inputs(model),
        "outputs": [item.name for item in model.graph.output],
        "node_count": len(model.graph.node),
        "operators": dict(sorted(operators.items())),
        "domains": dict(sorted(domains.items())),
        "expected_trt_plugins": plugin_nodes,
        "initializers_by_type": dict(sorted(initializers_by_type.items())),
        "matmul_int8_weight_input_count": len(set(matmul_int8_weight_inputs)),
        "matmul_dq_activation_input_count": len(set(matmul_dq_activation_inputs)),
        "quantization_scale_count": int(scale_values.size),
        "quantization_scales_finite": bool(
            scale_values.size > 0 and np.isfinite(scale_values).all())
        if operators.get("DequantizeLinear", 0) > 0
        else True,
        "quantization_scales_positive": bool(
            scale_values.size > 0 and (scale_values > 0).all())
        if operators.get("DequantizeLinear", 0) > 0
        else True,
    }


def summarize_calibration(path):
    with np.load(path) as calibration:
        keys = list(calibration.files)
        if keys != EXPECTED_INPUTS:
            raise AssertionError(
                f"Calibration keys do not match ONNX input order: {keys}"
            )
        first_dimension = calibration["prev_track_intances0"].shape[0]
        if first_dimension % 901 != 0:
            raise AssertionError(
                f"prev_track_intances0 first dimension {first_dimension} is not divisible by 901"
            )
        samples = first_dimension // 901
        if samples <= 0:
            raise AssertionError("Calibration contains no samples")
        digests = [hashlib.sha256() for _ in range(samples)]
        shapes = {}
        dtypes = {}
        for key in keys:
            array = calibration[key]
            shapes[key] = list(array.shape)
            dtypes[key] = str(array.dtype)
            per_frame = array.shape[0] // samples
            for sample_index, digest in enumerate(digests):
                chunk = np.ascontiguousarray(
                    array[
                        sample_index * per_frame:(sample_index + 1) * per_frame
                    ]
                )
                digest.update(key.encode("utf-8"))
                digest.update(chunk.tobytes())
        signatures = [digest.hexdigest() for digest in digests]
        unique_samples = len(set(signatures))
        if unique_samples != samples:
            raise AssertionError(
                f"Calibration contains only {unique_samples}/{samples} unique samples"
            )
    return {
        "path": str(path),
        "samples": samples,
        "unique_sample_signatures": unique_samples,
        "sample_sha256": signatures,
        "shapes": shapes,
        "dtypes": dtypes,
    }


def main():
    args = parse_args()
    fp_model = summarize_model(args.fp_onnx, run_checker=not args.skip_onnx_checker)
    quant_model = summarize_model(args.quant_onnx, run_checker=not args.skip_onnx_checker)
    calibration = summarize_calibration(args.calibration)

    if fp_model["inputs"] != EXPECTED_INPUTS:
        raise AssertionError(f"FP ONNX inputs differ from deployment contract: {fp_model['inputs']}")
    if quant_model["inputs"] != EXPECTED_INPUTS:
        raise AssertionError(
            f"Quantized ONNX inputs differ from deployment contract: {quant_model['inputs']}"
        )
    if quant_model["operators"].get("DequantizeLinear", 0) <= 0:
        raise AssertionError("Quantized ONNX has no DequantizeLinear nodes")
    if quant_model["initializers_by_type"].get("INT8", 0) <= 0:
        raise AssertionError("Quantized ONNX has no INT8 initializers")
    if quant_model["matmul_int8_weight_input_count"] != 0:
        raise AssertionError(
            "MatMul exclusion failed: one or more MatMul nodes consume INT8 weights"
        )
    if not quant_model["quantization_scales_finite"]:
        raise AssertionError("Quantized ONNX contains non-finite scales")
    if not quant_model["quantization_scales_positive"]:
        raise AssertionError("Quantized ONNX contains non-positive scales")

    result = {
        "schema_version": 1,
        "fp_onnx": fp_model,
        "quantized_onnx": quant_model,
        "calibration": calibration,
        "checks": {
            "onnx_checker": (
                "skipped_after_two_cpu_timeouts; TensorRT_10.7_parser_and_build_required"
                if args.skip_onnx_checker
                else "pass_with_expected_trt_plugins_assigned_temporary_domain"
            ),
            "deployment_input_contract": "pass",
            "explicit_dequantize_nodes": "pass",
            "int8_initializers": "pass",
            "matmul_weight_exclusion": "pass",
            "finite_positive_scales": "pass",
            "unique_calibration_samples": "pass",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["checks"], sort_keys=True))


if __name__ == "__main__":
    main()
