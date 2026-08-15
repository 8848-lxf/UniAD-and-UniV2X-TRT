#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


VARIANTS = ("baseline", "matmul_fp32", "mul_fp32", "matmul_mul_fp32")


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packed_occupancy(path):
    manifest = load_json(path.with_name(path.name + ".manifest.json"))
    frames = int(manifest["frames"])
    bit_count = int(np.prod(manifest["shape"], dtype=np.int64))
    packed_bytes = (bit_count + 7) // 8
    packed = np.fromfile(path, dtype=np.uint8).reshape(frames, packed_bytes)
    return packed, bit_count


def pairwise_occupancy(reference, actual, bit_count):
    reference_bits = np.unpackbits(
        reference, axis=1, bitorder="little")[:, :bit_count]
    actual_bits = np.unpackbits(
        actual, axis=1, bitorder="little")[:, :bit_count]
    intersection = int(np.logical_and(reference_bits, actual_bits).sum())
    union = int(np.logical_or(reference_bits, actual_bits).sum())
    return {
        "iou_percent": 100.0 * intersection / max(union, 1),
        "flipped_cells": int(np.not_equal(reference_bits, actual_bits).sum()),
        "byte_exact_frames": int(np.all(reference == actual, axis=1).sum()),
    }


def engine_layer_audit(path):
    layers = load_json(path)["Layers"]
    result = {}
    for operator in ("MatMul", "Mul"):
        marker = "ONNX Layer: %s_" % operator
        matched = [layer for layer in layers if marker in layer.get("Metadata", "")]
        float_only = 0
        half_present = 0
        for layer in matched:
            formats = {
                tensor.get("Format/Datatype")
                for tensor in layer.get("Inputs", []) + layer.get("Outputs", [])
            }
            if formats == {"Float"}:
                float_only += 1
            if "Half" in formats:
                half_present += 1
        result[operator] = {
            "engine_layers_with_onnx_metadata": len(matched),
            "float_only_io_layers": float_only,
            "layers_with_half_io": half_present,
            "note": (
                "Fused engine-layer I/O is an external audit; build-time OBEY "
                "constraints are authoritative for internal operator precision."
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    output = args.output or args.root / ("summary_%d.json" % args.frames)
    result_dir = args.root / ("probe%d" % args.frames)

    baseline_packed, bit_count = packed_occupancy(
        result_dir / "baseline" / "seg_out.packbits")
    rows = {}
    fixed_track_count = None
    for variant in VARIANTS:
        engine = (
            args.root / "engines" / ("uniad_tiny_fp16_%s.engine" % variant)
        )
        report = load_json(args.root / "reports" / (variant + ".json"))
        result_root = result_dir / variant
        occupancy = load_json(result_root / "occupancy_vs_pytorch.json")
        raw = load_json(result_root / "planning_metrics_raw.json")
        optimized = load_json(result_root / "planning_metrics_optimized.json")
        latency = load_json(result_root / "latency_metrics.json")
        variant_fixed_track_count = int(latency["fixed_track_input_count"])
        if fixed_track_count is None:
            fixed_track_count = variant_fixed_track_count
        elif fixed_track_count != variant_fixed_track_count:
            raise ValueError("Fixed track capacity mismatch for %s" % variant)
        packed, current_bit_count = packed_occupancy(
            result_root / "seg_out.packbits")
        if current_bit_count != bit_count or packed.shape != baseline_packed.shape:
            raise ValueError("Occupancy shape mismatch for %s" % variant)
        constraints = report.get("operator_kind_constraints")
        rows[variant] = {
            "engine": str(engine),
            "engine_sha256": sha256(engine),
            "build_seconds": report["build_seconds"],
            "track_profile": report["profile_shapes"]["prev_track_intances0"],
            "build_constraint_counts": (
                constraints["matched_counts"] if constraints else {}
            ),
            "engine_layer_audit": engine_layer_audit(
                args.root / "layer_info" / (variant + ".json")
            ),
            "occupancy_vs_pytorch": {
                key: occupancy[key]
                for key in (
                    "iou_percent", "precision_percent", "recall_percent",
                    "byte_exact_frames", "tensorrt_positive_cells",
                    "iou_by_horizon_percent",
                )
            },
            "occupancy_vs_baseline": pairwise_occupancy(
                baseline_packed, packed, bit_count),
            "raw": {
                "avg_l2_m": raw["avg_l2_m"],
                "box_col_percent": raw["avg_box_collision_percent"],
                "planning_mse_m2": raw.get("planning_mse"),
                "mean_point_l2_vs_pytorch_m": raw.get(
                    "planning_output_mean_point_l2_m"),
            },
            "optimized": {
                "avg_l2_m": optimized["avg_l2_m"],
                "box_col_percent": optimized["avg_box_collision_percent"],
            },
            "collision_optimizer": latency["collision_optimization_audit"],
            "latency_ms": {
                "model_mean": latency["model_enqueue"]["mean_ms"],
                "model_p50": latency["model_enqueue"]["p50_ms"],
                "model_p99": latency["model_enqueue"]["p99_ms"],
                "inference_mean": latency["inference_call"]["mean_ms"],
                "inference_p50": latency["inference_call"]["p50_ms"],
                "inference_p99": latency["inference_call"]["p99_ms"],
                "e2e_mean": latency["end_to_end"]["mean_ms"],
                "e2e_p50": latency["end_to_end"]["p50_ms"],
                "e2e_p99": latency["end_to_end"]["p99_ms"],
            },
        }

    baseline = rows["baseline"]
    for variant, row in rows.items():
        row["delta_vs_baseline"] = {
            "occupancy_iou_percentage_points": (
                row["occupancy_vs_pytorch"]["iou_percent"]
                - baseline["occupancy_vs_pytorch"]["iou_percent"]
            ),
            "optimized_box_col_percentage_points": (
                row["optimized"]["box_col_percent"]
                - baseline["optimized"]["box_col_percent"]
            ),
            "model_p50_percent": 100.0 * (
                row["latency_ms"]["model_p50"]
                / baseline["latency_ms"]["model_p50"] - 1.0
            ),
        }

    result = {
        "schema_version": 1,
        "experiment": "UniAD-tiny FP16 MatMul/Mul FP32 A/B",
        "frames": args.frames,
        "temporal_protocol": "official_literal",
        "fixed_track_count": fixed_track_count,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
