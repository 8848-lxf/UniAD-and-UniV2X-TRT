#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


PRECISIONS = ("fp32", "fp16", "int8_eq_fp16")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with path.open() as stream:
        return json.load(stream)


def filtered_stats(frame_csv, field, threshold_ms=1000.0):
    with frame_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [float(row[field]) for row in rows]
    retained = [value for value in values if value < threshold_ms]
    return {
        "total": len(values),
        "retained": len(retained),
        "threshold_ms": threshold_ms,
        "mean_ms": statistics.fmean(retained),
        "p50_ms": statistics.median(retained),
    }


def predictions_are_finite(path):
    count = 0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            count += 1
            for key, value in row.items():
                if key != "frame" and not math.isfinite(float(value)):
                    return count, False
    return count, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact = root / "artifacts" / "smoke_base_hybrid"

    conversion = load_json(artifact / "checkpoints" / "conversion_audit.json")
    quantization = load_json(artifact / "onnx" / "quantization_inspection.json")
    rows = {}
    for precision in PRECISIONS:
        evaluation = artifact / "evaluation" / f"tensorrt_{precision}"
        latency = load_json(evaluation / "latency_metrics.json")
        planning = load_json(evaluation / "planning_metrics.json")
        prediction_frames, finite = predictions_are_finite(
            evaluation / "planning_predictions.csv")
        rows[precision] = {
            "latency": latency,
            "planning": planning,
            "uncontended_model_diagnostic": filtered_stats(
                evaluation / "latency_metrics.json.frames.csv", "model_enqueue_ms"),
            "uncontended_e2e_diagnostic": filtered_stats(
                evaluation / "latency_metrics.json.frames.csv", "end_to_end_ms"),
            "prediction_frames": prediction_frames,
            "predictions_finite": finite,
        }
        if prediction_frames != 100 or not finite:
            raise AssertionError(
                f"{precision} predictions invalid: frames={prediction_frames}, finite={finite}"
            )

    files = [
        artifact / "checkpoints" / "uniad_base_to_tiny_hybrid_smoke.pth",
        artifact / "calibration" / "calib_data_shape0_901.npz",
        artifact / "onnx" / "uniad_tiny_imgx0.25_cp.repaired.onnx",
        artifact / "onnx" / "uniad_tiny_int8_eq_dq_only.onnx",
    ]
    files.extend((artifact / "engines").glob("*.engine"))
    files.extend((artifact / "timing_cache").glob("*.cache"))
    hashes = {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(files)
    }

    report = {
        "schema_version": 1,
        "status": "engineering_smoke_pass",
        "valid_for_official_accuracy_comparison": False,
        "invalidity_reason": (
            "The source is a UniAD-base checkpoint transplanted into the incompatible "
            "UniAD-tiny architecture; unmatched tensors retain deterministic initialization."
        ),
        "latency_warning": (
            "All GPUs were occupied by formal distributed training. Raw mean and percentile "
            "latencies contain multi-second scheduling stalls and are not benchmark results."
        ),
        "checkpoint_conversion": conversion,
        "quantization_checks": quantization["checks"],
        "quantization_summary": {
            "calibration_samples": quantization["calibration"]["samples"],
            "dequantize_nodes": quantization["quantized_onnx"]["operators"].get(
                "DequantizeLinear", 0),
            "int8_initializers": quantization["quantized_onnx"]["initializers_by_type"].get(
                "INT8", 0),
            "matmul_int8_weight_inputs": quantization["quantized_onnx"][
                "matmul_int8_weight_input_count"],
        },
        "results": rows,
        "artifacts": hashes,
    }
    reports = artifact / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "smoke_report.json").open("w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")

    labels = {
        "fp32": "TensorRT 10.9 FP32",
        "fp16": "TensorRT 10.9 FP16",
        "int8_eq_fp16": "TensorRT 10.9 INT8(EQ)+FP16",
    }
    lines = [
        "# UniAD base-to-tiny hybrid engineering smoke",
        "",
        "Status: **PASS for deployment-chain mechanics only**.",
        "",
        "This result is not valid for comparison with NVIDIA's UniAD-tiny accuracy table. "
        "The source checkpoint is UniAD-base (R101, 200x200 BEV), while the deployment "
        "target is UniAD-tiny (R50, 50x50 BEV).",
        "",
        "## Checkpoint conversion",
        "",
        f"- Transferred keys: {conversion['transferred_keys']}/{conversion['target_keys']}",
        f"- Transferred parameter fraction: {conversion['transferred_numel_fraction']:.6%}",
        f"- Shape conflicts: {len(conversion['shape_mismatches'])}",
        f"- Unexpected source keys: {len(conversion['unexpected_source_keys'])}",
        "",
        "## Quantization gates",
        "",
        f"- Calibration samples: {report['quantization_summary']['calibration_samples']} (smoke only)",
        f"- DequantizeLinear nodes: {report['quantization_summary']['dequantize_nodes']}",
        f"- INT8 initializers: {report['quantization_summary']['int8_initializers']}",
        f"- MatMul INT8 weight inputs: {report['quantization_summary']['matmul_int8_weight_inputs']}",
        "",
        "## 100-frame smoke results",
        "",
        "Raw mean and p50 are retained as requested, but are invalid as benchmarks because "
        "formal 8-GPU training caused multi-second scheduling stalls. The filtered columns "
        "are diagnostics only and exclude samples >=1000 ms.",
        "",
        "| Engine | model mean ms | model p50 ms | E2E mean ms | E2E p50 ms | <1s model mean/p50 ms | retained | avg L2 m | box Col % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for precision in PRECISIONS:
        item = rows[precision]
        model = item["latency"]["model_enqueue"]
        e2e = item["latency"]["end_to_end"]
        diagnostic = item["uncontended_model_diagnostic"]
        planning = item["planning"]
        lines.append(
            f"| {labels[precision]} | {model['mean_ms']:.3f} | {model['p50_ms']:.3f} | "
            f"{e2e['mean_ms']:.3f} | {e2e['p50_ms']:.3f} | "
            f"{diagnostic['mean_ms']:.3f}/{diagnostic['p50_ms']:.3f} | "
            f"{diagnostic['retained']}/{diagnostic['total']} | "
            f"{planning['avg_l2_m']:.6f} | {planning['avg_box_collision_percent']:.6f} |"
        )
    lines.extend([
        "",
        "All three engines produced 100 finite planning trajectories. Accuracy values above "
        "are diagnostic outputs from the hybrid checkpoint and must not be interpreted as "
        "UniAD-tiny reproduction results.",
        "",
        "## Artifacts",
        "",
    ])
    for path, info in hashes.items():
        lines.append(f"- `{path}`: {info['bytes']} bytes, SHA256 `{info['sha256']}`")
    (reports / "smoke_report.md").write_text("\n".join(lines) + "\n")
    print(reports / "smoke_report.md")


if __name__ == "__main__":
    main()
