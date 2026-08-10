import argparse
import json
import os
import time

import numpy as np
import torch

from trt_engine import TensorRTEngine
from trt_runtime import EGO_INPUT_NAMES, INPUT_NAMES, load_export_inputs


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def summarize_output(value):
    result = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "all_finite": True,
        "nonfinite_count": 0,
    }
    if value.is_floating_point():
        finite = torch.isfinite(value)
        result["nonfinite_count"] = int((~finite).sum().item())
        result["all_finite"] = result["nonfinite_count"] == 0
        if finite.any():
            finite_values = value[finite]
            result["finite_min"] = float(finite_values.min().item())
            result["finite_max"] = float(finite_values.max().item())
        else:
            result["finite_min"] = None
            result["finite_max"] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("plugin")
    parser.add_argument("input_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--coop-input-npz")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--context-cache-size", type=int, default=1)
    args = parser.parse_args()

    engine = TensorRTEngine(
        args.engine, args.plugin, context_cache_size=args.context_cache_size
    )
    values = load_export_inputs(
        args.input_npz, torch.device("cuda"), args.coop_input_npz
    )
    input_names = EGO_INPUT_NAMES if args.coop_input_npz else INPUT_NAMES
    input_map = dict(zip(input_names, values))
    execution_stream = torch.cuda.Stream()
    rows = []
    outputs = None
    for index in range(args.warmup + args.repeats):
        execution_stream.synchronize()
        with torch.cuda.stream(execution_stream):
            start = time.perf_counter()
            outputs = engine.infer(input_map, synchronize=False)
            enqueued = time.perf_counter()
        execution_stream.synchronize()
        completed = time.perf_counter()
        rows.append({
            "iteration": index,
            "warmup": index < args.warmup,
            "enqueue_ms": (enqueued - start) * 1000.0,
            "completion_wait_ms": (completed - enqueued) * 1000.0,
            "total_ms": (completed - start) * 1000.0,
        })
    measured = rows[args.warmup:]
    result = {
        "engine": os.path.abspath(args.engine),
        "gpu": torch.cuda.get_device_name(0),
        "fixed_input_repeats": args.repeats,
        "warmup": args.warmup,
        "latency": {
            key: summarize([row[key] for row in measured])
            for key in ("enqueue_ms", "completion_wait_ms", "total_ms")
        },
        "rows": rows,
        "runtime_output_buffers": engine.allocation_stats(),
        "contract": engine.describe(),
        "outputs": {
            name: summarize_output(value) for name, value in outputs.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
