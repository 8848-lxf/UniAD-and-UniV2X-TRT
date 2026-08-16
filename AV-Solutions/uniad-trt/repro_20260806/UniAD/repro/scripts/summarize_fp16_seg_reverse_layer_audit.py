#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PROBE_SETS = {
    "coarse": [
        ("p00_input1743", "input.1743"),
        ("p01_state", "state"),
        ("p02_layer0_out", "input.1807"),
        ("p03_layer0_fused", "input.1819"),
        ("p04_layer1_out", "input.1883"),
        ("p05_layer1_fused", "input.1895"),
        ("p06_layer2_out", "input.1959"),
        ("p07_layer2_fused", "input.1971"),
        ("p08_layer3_out", "input.2035"),
        ("p09_layer3_fused", "input.2047"),
        ("p10_layer4_out", "input.2111"),
        ("p11_layer4_fused", "onnx::Unsqueeze_26264"),
        ("p12_dense_input", "input.2123"),
        ("p13_dense_stage0", "input.2143"),
        ("p14_dense_stage1", "future_states"),
        ("p15_dense_crop", "future_states.3"),
        ("p16_occ_einsum", "onnx::Slice_26434"),
        ("p17_occ_slice", "onnx::Sigmoid_26439"),
        ("p18_occ_sigmoid", "onnx::Mul_26440"),
        ("p19_occ_mul", "pred_ins_sigmoid"),
        ("p20_occ_concat", "onnx::ReduceMax_26477"),
        ("p21_occ_score", "onnx::Greater_26478"),
    ],
    "layer0": [
        ("f00_state", "state"),
        ("f01_gate_logits", "onnx::Sigmoid_23764"),
        ("f02_gate_score", "onnx::Less_23765"),
        ("f03_query", "query.135"),
        ("f04_self_q", "q.107"),
        ("f05_self_logits", "attn.107"),
        ("f06_self_softmax", "onnx::MatMul_23950"),
        ("f07_self_residual", "input.1783"),
        ("f08_norm0", "query.139"),
        ("f09_cross_q", "q.111"),
        ("f10_cross_logits", "attn.111"),
        ("f11_cross_softmax", "onnx::MatMul_24104"),
        ("f12_cross_residual", "input.1791"),
        ("f13_norm1", "x.231"),
        ("f14_ffn_hidden", "input.1795"),
        ("f15_ffn_relu", "onnx::MatMul_24141"),
        ("f16_ffn_output", "input.1799"),
        ("f17_ffn_residual", "input.1803"),
        ("f18_norm2", "onnx::Add_24155"),
        ("f19_layer0_out", "input.1807"),
    ],
    "upstream": [
        ("u00_track_scores", "track_scores"),
        ("u01_active_index", "active_index.7"),
        ("u02_valid_mask", "onnx::Greater_23426"),
        ("u03_nonzero", "onnx::NonZero_23428"),
        ("u04_out_track_query", "out_track_query"),
        ("u05_ins_embed", "ins_embed"),
        ("u06_scores1", "scores.1"),
        ("u07_track_sigmoid_logits", "onnx::Sigmoid_10611"),
        ("u08_track_sigmoid", "onnx::ReduceMax_10612"),
    ],
}


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def load_manifest(path, expected_frames):
    data = json.loads(path.read_text())
    if data.get("dtype") != "float32":
        raise ValueError(f"Unexpected dtype in {path}: {data.get('dtype')}")
    if data.get("frames") != expected_frames:
        raise ValueError(
            f"Unexpected frame count in {path}: {data.get('frames')}")
    if "shapes_per_frame" in data:
        shapes = [
            tuple(int(dim) for dim in shape)
            for shape in data["shapes_per_frame"]
        ]
    else:
        shape = tuple(int(dim) for dim in data["shape_per_frame"])
        shapes = [shape] * expected_frames
    if len(shapes) != expected_frames:
        raise ValueError(
            f"Unexpected shape count in {path}: {len(shapes)}")
    if any(any(dim < 0 for dim in shape) for shape in shapes):
        raise ValueError(f"Negative audit shape in {path}")
    return data, shapes


def shape_histogram(shapes):
    counts = {}
    for shape in shapes:
        key = "x".join(str(dim) for dim in shape)
        counts[key] = counts.get(key, 0) + 1
    return counts


def compare_probe(root, key, tensor, frames):
    probe_root = root / f"probe{frames}"
    arrays = {}
    manifests = {}
    shapes_per_precision = {}
    offsets = {}
    for precision in ("fp32", "fp16"):
        result_root = probe_root / f"{key}_{precision}"
        manifest_path = result_root / "audit_out.float32.manifest.json"
        output_path = result_root / "audit_out.float32"
        manifest, shapes = load_manifest(manifest_path, frames)
        frame_counts = [math.prod(shape) for shape in shapes]
        count = sum(frame_counts)
        expected_bytes = count * np.dtype(np.float32).itemsize
        actual_bytes = output_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Unexpected byte count for {output_path}: "
                f"{actual_bytes} != {expected_bytes}")
        arrays[precision] = np.memmap(
            output_path, dtype=np.float32, mode="r", shape=(count,))
        manifests[precision] = manifest
        shapes_per_precision[precision] = shapes
        offsets[precision] = np.cumsum([0] + frame_counts)

    sum_abs = 0.0
    sum_sq = 0.0
    sum_ref_sq = 0.0
    sum_ref_abs = 0.0
    sum_dot = 0.0
    sum_fp16_sq = 0.0
    element_count = 0
    max_abs = 0.0
    nonfinite_fp32 = 0
    nonfinite_fp16 = 0
    shape_mismatch_frames = []
    aligned_empty_frames = []
    frame_metrics = []
    for frame in range(frames):
        fp32_shape = shapes_per_precision["fp32"][frame]
        fp16_shape = shapes_per_precision["fp16"][frame]
        if fp32_shape != fp16_shape:
            shape_mismatch_frames.append({
                "frame": frame,
                "fp32_shape": list(fp32_shape),
                "fp16_shape": list(fp16_shape),
            })
            continue
        if math.prod(fp32_shape) == 0:
            aligned_empty_frames.append(frame)
            continue
        ref = np.asarray(
            arrays["fp32"][offsets["fp32"][frame]:offsets["fp32"][frame + 1]],
            dtype=np.float64)
        test = np.asarray(
            arrays["fp16"][offsets["fp16"][frame]:offsets["fp16"][frame + 1]],
            dtype=np.float64)
        nonfinite_fp32 += int((~np.isfinite(ref)).sum())
        nonfinite_fp16 += int((~np.isfinite(test)).sum())
        finite = np.isfinite(ref) & np.isfinite(test)
        ref = ref[finite]
        test = test[finite]
        if not ref.size:
            raise ValueError(f"No finite values for {key}, frame {frame}")
        diff = test - ref
        frame_sq = float(np.dot(diff, diff))
        frame_ref_sq = float(np.dot(ref, ref))
        frame_test_sq = float(np.dot(test, test))
        frame_dot = float(np.dot(ref, test))
        frame_abs = float(np.abs(diff).sum())
        frame_max_abs = float(np.abs(diff).max())
        frame_rmse = math.sqrt(frame_sq / ref.size)
        frame_ref_rms = math.sqrt(frame_ref_sq / ref.size)
        frame_relative_rmse = (
            100.0 * math.sqrt(frame_sq / frame_ref_sq)
            if frame_ref_sq > 0.0 else math.inf)
        frame_cosine = (
            frame_dot / math.sqrt(frame_ref_sq * frame_test_sq)
            if frame_ref_sq > 0.0 and frame_test_sq > 0.0 else 0.0)
        frame_metrics.append({
            "frame": frame,
            "mae": frame_abs / ref.size,
            "rmse": frame_rmse,
            "reference_rms": frame_ref_rms,
            "relative_rmse_percent": frame_relative_rmse,
            "cosine": frame_cosine,
            "max_abs": frame_max_abs,
        })
        sum_abs += frame_abs
        sum_sq += frame_sq
        sum_ref_sq += frame_ref_sq
        sum_ref_abs += float(np.abs(ref).sum())
        sum_dot += frame_dot
        sum_fp16_sq += frame_test_sq
        element_count += ref.size
        max_abs = max(max_abs, frame_max_abs)

    if element_count == 0:
        raise ValueError(f"No shape-aligned values for {key}")
    relative_rmse = (
        100.0 * math.sqrt(sum_sq / sum_ref_sq)
        if sum_ref_sq > 0.0 else math.inf)
    cosine = (
        sum_dot / math.sqrt(sum_ref_sq * sum_fp16_sq)
        if sum_ref_sq > 0.0 and sum_fp16_sq > 0.0 else 0.0)
    frame_relative = [item["relative_rmse_percent"] for item in frame_metrics]
    return {
        "key": key,
        "tensor": tensor,
        "fp32_shape_histogram": shape_histogram(shapes_per_precision["fp32"]),
        "fp16_shape_histogram": shape_histogram(shapes_per_precision["fp16"]),
        "shape_aligned_frames": frames - len(shape_mismatch_frames),
        "nonempty_shape_aligned_frames": len(frame_metrics),
        "aligned_empty_frames": aligned_empty_frames,
        "shape_mismatch_frame_count": len(shape_mismatch_frames),
        "shape_mismatch_frames": shape_mismatch_frames,
        "elements_compared": element_count,
        "nonfinite_fp32": nonfinite_fp32,
        "nonfinite_fp16": nonfinite_fp16,
        "global": {
            "mae": sum_abs / element_count,
            "rmse": math.sqrt(sum_sq / element_count),
            "reference_rms": math.sqrt(sum_ref_sq / element_count),
            "relative_rmse_percent": relative_rmse,
            "normalized_mae_percent": (
                100.0 * sum_abs / sum_ref_abs if sum_ref_abs > 0.0 else math.inf),
            "cosine": cosine,
            "max_abs": max_abs,
        },
        "per_frame_relative_rmse_percent": {
            "mean": float(np.mean(frame_relative)),
            "p50": percentile(frame_relative, 50),
            "p90": percentile(frame_relative, 90),
            "p99": percentile(frame_relative, 99),
            "max": max(frame_relative),
        },
        "per_frame": frame_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=sorted(PROBE_SETS), default="coarse")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    results = [
        compare_probe(args.root, key, tensor, args.frames)
        for key, tensor in PROBE_SETS[args.stage]
    ]
    previous = None
    for result in results:
        current = result["global"]["relative_rmse_percent"]
        result["relative_rmse_delta_from_previous_pp"] = (
            None if previous is None else current - previous)
        result["relative_rmse_ratio_to_previous"] = (
            None if previous in (None, 0.0) else current / previous)
        previous = current

    payload = {
        "schema_version": 1,
        "frames": args.frames,
        "stage": args.stage,
        "input_protocol": "identical recorded inputs; recurrent feedback disabled",
        "precision_pair": ["TensorRT FP32", "TensorRT FP16"],
        "probe_order": "forward from occupancy shared feature to seg_out score",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    fieldnames = [
        "key", "tensor", "shape", "mae", "rmse", "reference_rms",
        "relative_rmse_percent", "normalized_mae_percent", "cosine",
        "max_abs", "frame_rrmse_p50", "frame_rrmse_p99",
        "delta_from_previous_pp", "ratio_to_previous",
    ]
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            global_metrics = result["global"]
            frame_metrics = result["per_frame_relative_rmse_percent"]
            writer.writerow({
                "key": result["key"],
                "tensor": result["tensor"],
                "shape": json.dumps({
                    "fp32": result["fp32_shape_histogram"],
                    "fp16": result["fp16_shape_histogram"],
                }, sort_keys=True),
                "mae": global_metrics["mae"],
                "rmse": global_metrics["rmse"],
                "reference_rms": global_metrics["reference_rms"],
                "relative_rmse_percent": global_metrics["relative_rmse_percent"],
                "normalized_mae_percent": global_metrics["normalized_mae_percent"],
                "cosine": global_metrics["cosine"],
                "max_abs": global_metrics["max_abs"],
                "frame_rrmse_p50": frame_metrics["p50"],
                "frame_rrmse_p99": frame_metrics["p99"],
                "delta_from_previous_pp": result["relative_rmse_delta_from_previous_pp"],
                "ratio_to_previous": result["relative_rmse_ratio_to_previous"],
            })


if __name__ == "__main__":
    main()
