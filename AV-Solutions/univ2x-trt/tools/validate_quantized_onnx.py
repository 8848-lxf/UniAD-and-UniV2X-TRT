import argparse
import collections
import copy
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


PLUGIN_OPS = {
    "InverseTRT",
    "ModulatedDeformableConv2dTRT",
    "MultiScaleDeformableAttnTRT",
    "RotateTRT",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_inputs(model):
    initializers = {item.name for item in model.graph.initializer}
    return [item.name for item in model.graph.input if item.name not in initializers]


def checker_with_plugin_domains(model):
    checkable = copy.deepcopy(model)
    plugin_count = 0
    for node in checkable.graph.node:
        if node.op_type in PLUGIN_OPS:
            node.domain = "nvidia.trt.plugin"
            plugin_count += 1
    if plugin_count and not any(
        item.domain == "nvidia.trt.plugin" for item in checkable.opset_import
    ):
        checkable.opset_import.add(domain="nvidia.trt.plugin", version=1)
    onnx.checker.check_model(checkable, check_custom_domain=False)
    return plugin_count


def summarize_model(path):
    model = onnx.load(path, load_external_data=True)
    plugin_count = checker_with_plugin_domains(model)
    operators = collections.Counter(node.op_type for node in model.graph.node)
    initializers = {item.name: item for item in model.graph.initializer}
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    scales = []
    matmul_int8_weights = set()
    matmul_dq_activations = set()
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            if len(node.input) > 1 and node.input[1] in initializers:
                scales.append(
                    numpy_helper.to_array(initializers[node.input[1]])
                    .astype(np.float64).reshape(-1)
                )
        if node.op_type != "MatMul":
            continue
        for input_name in node.input:
            producer = producers.get(input_name)
            if producer is None or producer.op_type != "DequantizeLinear":
                continue
            source = initializers.get(producer.input[0])
            if source is not None and source.data_type == TensorProto.INT8:
                matmul_int8_weights.add(node.name)
            else:
                matmul_dq_activations.add(node.name)
    scale_values = np.concatenate(scales) if scales else np.asarray([])
    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "sha256": sha256(path),
        "nodes": len(model.graph.node),
        "inputs": graph_inputs(model),
        "outputs": [item.name for item in model.graph.output],
        "operators": dict(sorted(operators.items())),
        "plugin_nodes": plugin_count,
        "initializer_types": dict(sorted(collections.Counter(
            TensorProto.DataType.Name(item.data_type)
            for item in model.graph.initializer
        ).items())),
        "scale_count": int(scale_values.size),
        "scales_finite": bool(scale_values.size and np.isfinite(scale_values).all()),
        "scales_positive": bool(scale_values.size and (scale_values > 0).all()),
        "scale_min": float(scale_values.min()) if scale_values.size else None,
        "scale_max": float(scale_values.max()) if scale_values.size else None,
        "matmul_int8_weight_count": len(matmul_int8_weights),
        "matmul_dq_activation_count": len(matmul_dq_activations),
    }


def summarize_calibration(path, report_path):
    with open(report_path) as handle:
        report = json.load(handle)
    single_shapes = report["output"]["single_shapes"]
    sample_count = report["output"]["samples"]
    with np.load(path) as archive:
        nonfinite = {}
        shape_errors = {}
        for name in archive.files:
            value = archive[name]
            expected = list(single_shapes[name])
            expected[0] *= sample_count
            if list(value.shape) != expected:
                shape_errors[name] = {
                    "actual": list(value.shape), "expected": expected
                }
            if np.issubdtype(value.dtype, np.floating):
                count = int(value.size - np.count_nonzero(np.isfinite(value)))
                if count:
                    nonfinite[name] = count
        keys = list(archive.files)
    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "sha256": sha256(path),
        "samples": sample_count,
        "keys": keys,
        "shape_errors": shape_errors,
        "nonfinite": nonfinite,
        "use_prev_bev": report["output"]["use_prev_bev"],
        "commands": report["output"]["commands"],
        "dataset_split": report.get("dataset_split"),
        "files_valid": True,
        "sample_count_valid": True,
        "total_bytes_valid": True,
        "dtype_errors": {},
        "dataset_indices_valid": True,
    }


def summarize_calibration_manifest(path):
    manifest_path = os.path.abspath(path)
    manifest_dir = os.path.dirname(manifest_path)
    with open(manifest_path) as handle:
        manifest = json.load(handle)

    missing_files = []
    size_errors = {}
    shape_errors = {}
    dtype_errors = {}
    nonfinite = {}
    sample_hashes = {}
    observed_bytes = 0
    use_prev_bev = []
    commands = []
    track_counts = []
    coop_track_counts = []
    dataset_indices = []
    for sample in manifest.get("samples", []):
        filename = sample["file"]
        sample_path = os.path.join(manifest_dir, filename)
        if not os.path.isfile(sample_path):
            missing_files.append(filename)
            continue
        actual_bytes = os.path.getsize(sample_path)
        observed_bytes += actual_bytes
        if actual_bytes != sample.get("bytes"):
            size_errors[filename] = {
                "actual": actual_bytes,
                "expected": sample.get("bytes"),
            }
        sample_hashes[filename] = sha256(sample_path)
        with np.load(sample_path) as archive:
            expected_keys = set(sample.get("shapes", {}))
            actual_keys = set(archive.files)
            if actual_keys != expected_keys:
                shape_errors[filename + ":keys"] = {
                    "actual": sorted(actual_keys),
                    "expected": sorted(expected_keys),
                }
            for name in archive.files:
                value = archive[name]
                expected_shape = sample.get("shapes", {}).get(name)
                if expected_shape is None or list(value.shape) != expected_shape:
                    shape_errors[filename + ":" + name] = {
                        "actual": list(value.shape),
                        "expected": expected_shape,
                    }
                expected_dtype = sample.get("dtypes", {}).get(name)
                if expected_dtype is None or str(value.dtype) != expected_dtype:
                    dtype_errors[filename + ":" + name] = {
                        "actual": str(value.dtype),
                        "expected": expected_dtype,
                    }
                if np.issubdtype(value.dtype, np.floating):
                    count = int(value.size - np.count_nonzero(np.isfinite(value)))
                    if count:
                        nonfinite[filename + ":" + name] = count
        dataset_indices.append(sample.get("dataset_index"))
        use_prev_bev.append(sample.get("use_prev_bev"))
        commands.append(sample.get("command"))
        track_counts.append(sample.get("track_count"))
        if sample.get("coop_track_count") is not None:
            coop_track_counts.append(sample["coop_track_count"])

    sample_count = len(manifest.get("samples", []))
    selected_indices = set(manifest.get("selected_indices", []))
    return {
        "path": manifest_path,
        "bytes": os.path.getsize(manifest_path),
        "sha256": sha256(manifest_path),
        "dataset_split": manifest.get("dataset_split"),
        "samples": sample_count,
        "selected_index_count": len(selected_indices),
        "dataset_indices_valid": all(
            index in selected_indices for index in dataset_indices
        ),
        "files_valid": not missing_files and not size_errors,
        "sample_count_valid": sample_count == manifest.get("sample_count"),
        "total_bytes_valid": observed_bytes == manifest.get("total_bytes"),
        "missing_files": missing_files,
        "size_errors": size_errors,
        "shape_errors": shape_errors,
        "dtype_errors": dtype_errors,
        "nonfinite": nonfinite,
        "sample_hashes": sample_hashes,
        "use_prev_bev": use_prev_bev,
        "commands": commands,
        "track_count_range": (
            [min(track_counts), max(track_counts)] if track_counts else None
        ),
        "coop_track_count_range": (
            [min(coop_track_counts), max(coop_track_counts)]
            if coop_track_counts else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-onnx", required=True)
    parser.add_argument("--quant-onnx", required=True)
    calibration_group = parser.add_mutually_exclusive_group(required=True)
    calibration_group.add_argument("--calibration")
    calibration_group.add_argument("--calibration-manifest")
    parser.add_argument("--calibration-report")
    parser.add_argument("--quantization-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fp_model = summarize_model(args.fp_onnx)
    quant_model = summarize_model(args.quant_onnx)
    if args.calibration:
        if not args.calibration_report:
            parser.error("--calibration requires --calibration-report")
        calibration = summarize_calibration(
            args.calibration, args.calibration_report
        )
    else:
        calibration = summarize_calibration_manifest(
            args.calibration_manifest
        )
    with open(args.quantization_report) as handle:
        quantization = json.load(handle)
    checks = {
        "onnx_checker": True,
        "input_contract": fp_model["inputs"] == quant_model["inputs"],
        "output_contract": fp_model["outputs"] == quant_model["outputs"],
        "plugin_nodes_present": quant_model["plugin_nodes"] > 0,
        "quantize_nodes_present": quant_model["operators"].get("QuantizeLinear", 0) > 0,
        "dequantize_nodes_present": quant_model["operators"].get("DequantizeLinear", 0) > 0,
        "int8_initializers_present": quant_model["initializer_types"].get("INT8", 0) > 0,
        "matmul_weight_exclusion": quant_model["matmul_int8_weight_count"] == 0,
        "finite_positive_scales": (
            quant_model["scales_finite"] and quant_model["scales_positive"]
        ),
        "calibration_shapes": not calibration["shape_errors"],
        "calibration_dtypes": not calibration["dtype_errors"],
        "calibration_finite": not calibration["nonfinite"],
        "calibration_files": calibration["files_valid"],
        "calibration_sample_count": calibration["sample_count_valid"],
        "calibration_total_bytes": calibration["total_bytes_valid"],
        "calibration_dataset_indices": calibration["dataset_indices_valid"],
        "calibration_train_split": calibration["dataset_split"] == "train",
        "quantization_output_hash": (
            quantization["output"]["sha256"] == quant_model["sha256"]
        ),
        "calibration_hash": (
            quantization["inputs"]["calibration_sha256"]
            == calibration["sha256"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "checks": checks,
        "failed": failed,
        "fp_onnx": fp_model,
        "quantized_onnx": quant_model,
        "calibration": calibration,
        "quantization_report": os.path.abspath(args.quantization_report),
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"checks": checks, "failed": failed}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
