#!/usr/bin/env python3

"""Compare a TensorRT engine with a saved PyTorch output on identical inputs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--raw-input-dir", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--trt-engine-module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_shape(shape, element_count, name):
    shape = tuple(int(value) for value in shape)
    dynamic_axes = [index for index, value in enumerate(shape) if value < 0]
    if not dynamic_axes:
        expected = int(np.prod(shape, dtype=np.int64))
        if expected != element_count:
            raise ValueError(
                f"{name}: file has {element_count} elements, expected {expected}"
            )
        return shape
    if len(dynamic_axes) != 1:
        raise ValueError(f"{name}: unsupported shape with multiple dynamic axes: {shape}")
    static_count = int(
        np.prod([value for value in shape if value >= 0], dtype=np.int64)
    )
    if static_count == 0 or element_count % static_count:
        raise ValueError(
            f"{name}: cannot infer {shape} from {element_count} elements"
        )
    result = list(shape)
    result[dynamic_axes[0]] = element_count // static_count
    return tuple(result)


def main():
    args = parse_args()
    sys.path.insert(0, str(args.trt_engine_module.resolve()))
    from trt_engine import TensorRTEngine

    runner = TensorRTEngine(args.engine, args.plugin)
    inputs = {}
    input_manifest = {}
    for name in runner.input_names:
        path = args.raw_input_dir / f"{name}.dat"
        if not path.is_file():
            raise FileNotFoundError(f"missing raw input: {path}")
        dtype = np.dtype(trt.nptype(runner.engine.get_tensor_dtype(name)))
        flat = np.fromfile(path, dtype=dtype)
        shape = resolved_shape(runner.engine.get_tensor_shape(name), flat.size, name)
        inputs[name] = flat.reshape(shape)
        input_manifest[name] = {
            "dtype": str(dtype),
            "shape": list(shape),
            "sha256": sha256(path),
        }

    outputs = runner.infer(inputs)
    if "outs_planning" not in outputs:
        raise KeyError(f"engine has no outs_planning output: {runner.output_names}")
    actual = outputs["outs_planning"].detach().cpu().numpy().astype(np.float64)
    reference = np.load(args.reference).astype(np.float64)
    if actual.shape != reference.shape:
        raise ValueError(
            f"planning shape mismatch: TensorRT {actual.shape}, reference {reference.shape}"
        )

    delta = actual - reference
    point_l2 = np.linalg.norm(delta, axis=-1)
    squared_point_l2 = np.square(delta).sum(axis=-1)
    output_manifest = {}
    for name, tensor in outputs.items():
        value = tensor.detach().cpu().numpy()
        finite = np.isfinite(value) if np.issubdtype(value.dtype, np.number) else None
        output_manifest[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "nonzero_count": int(np.count_nonzero(value)),
            "finite_count": int(finite.sum()) if finite is not None else None,
            "min": float(value.min()) if value.size else None,
            "max": float(value.max()) if value.size else None,
        }
    result = {
        "schema_version": 1,
        "comparison": "TensorRT and PyTorch outputs on byte-identical saved inputs",
        "engine": str(args.engine.resolve()),
        "engine_sha256": sha256(args.engine),
        "plugin": str(args.plugin.resolve()),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "input_manifest": input_manifest,
        "output_manifest": output_manifest,
        "outs_planning_shape": list(actual.shape),
        "mean_point_l2_m": float(point_l2.mean()),
        "coordinate_mse_m2": float(np.square(delta).mean()),
        "mean_squared_point_l2_m2": float(squared_point_l2.mean()),
        "root_mean_squared_point_l2_m": float(np.sqrt(squared_point_l2.mean())),
        "max_point_l2_m": float(point_l2.max()),
        "max_abs_coordinate_delta_m": float(np.abs(delta).max()),
        "tensorrt_outs_planning": actual.tolist(),
        "pytorch_outs_planning": reference.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
