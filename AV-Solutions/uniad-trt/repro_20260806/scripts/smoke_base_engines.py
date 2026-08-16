#!/usr/bin/env python3

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[5]
RUNTIME_TOOLS = WORKSPACE / "UniV2X" / "deploy_int8" / "tools"
sys.path.insert(0, str(RUNTIME_TOOLS))

from trt_engine import TensorRTEngine  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def split_sample(calibration, names, sample_index, sample_count):
    result = {}
    for name in names:
        value = calibration[name]
        if value.shape[0] % sample_count:
            raise ValueError(
                "%s leading dimension %d is not divisible by %d"
                % (name, value.shape[0], sample_count)
            )
        stride = value.shape[0] // sample_count
        begin = sample_index * stride
        result[name] = np.ascontiguousarray(value[begin:begin + stride])
    return result


def output_nonfinite(outputs):
    result = {}
    for name, value in outputs.items():
        result[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nan": int(torch.isnan(value).sum().item())
            if value.is_floating_point() else 0,
            "positive_inf": int(torch.isposinf(value).sum().item())
            if value.is_floating_point() else 0,
            "negative_inf": int(torch.isneginf(value).sum().item())
            if value.is_floating_point() else 0,
        }
    return result


def benchmark(name, engine_path, plugin_path, arrays, warmup, iterations):
    runtime = TensorRTEngine(engine_path, plugin_path, context_cache_size=1)
    inputs = {
        tensor_name: torch.as_tensor(arrays[tensor_name]).to(
            device="cuda", dtype=runtime.dtypes[tensor_name]
        )
        for tensor_name in runtime.input_names
    }
    for _ in range(warmup):
        runtime.infer(inputs)

    gpu_ms = []
    wall_ms = []
    outputs = None
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        outputs = runtime.infer(inputs, synchronize=False)
        end_event.record()
        end_event.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
        gpu_ms.append(float(start_event.elapsed_time(end_event)))

    result = {
        "name": name,
        "engine": os.path.realpath(engine_path),
        "engine_bytes": os.path.getsize(engine_path),
        "forward_gpu": summarize(gpu_ms),
        "runtime_call_end_to_end": summarize(wall_ms),
        "outputs": output_nonfinite(outputs),
        "allocation_stats": runtime.allocation_stats(),
    }
    del outputs, inputs, runtime
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    engines = []
    for spec in args.engine:
        name, separator, path = spec.partition("=")
        if not separator:
            raise ValueError("Expected --engine NAME=PATH, got %r" % spec)
        engines.append((name, os.path.realpath(path)))

    with np.load(args.calibration) as calibration:
        sample_count = calibration["prev_track_intances0"].shape[0] // 901
        if not 0 <= args.sample_index < sample_count:
            raise ValueError("sample-index is outside calibration sample range")
        first_runtime = TensorRTEngine(engines[0][1], args.plugin)
        input_names = list(first_runtime.input_names)
        del first_runtime
        torch.cuda.empty_cache()
        arrays = split_sample(
            calibration, input_names, args.sample_index, sample_count
        )

    report = {
        "calibration": os.path.realpath(args.calibration),
        "calibration_samples": sample_count,
        "sample_index": args.sample_index,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "gpu": torch.cuda.get_device_name(),
        "results": [
            benchmark(name, path, args.plugin, arrays, args.warmup, args.iterations)
            for name, path in engines
        ],
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
