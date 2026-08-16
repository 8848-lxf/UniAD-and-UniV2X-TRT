#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

OFFICIAL = {
    "pytorch_fp32": {"model_ms": 843.5172, "fps": 1.18, "l2": 0.9986, "col": 0.27, "planning": 0.0},
    "tensorrt_fp32": {"model_ms": 64.0469, "fps": 15.61, "l2": 0.9986, "col": 0.27, "planning": 9.2417e-7},
    "tensorrt_fp16": {"model_ms": 49.7559, "fps": 20.10, "l2": 1.0021, "col": 0.26, "planning": 0.0458},
    "tensorrt_int8_eq_fp16": {"model_ms": 39.3125, "fps": 25.44, "l2": 1.0029, "col": 0.27, "planning": 0.0502},
}

OFFICIAL_LABELS = {
    "pytorch_fp32": "PyTorch 1.12 FP32",
    "tensorrt_fp32": "TensorRT 10.7 FP32",
    "tensorrt_fp16": "TensorRT 10.7 FP16",
    "tensorrt_int8_eq_fp16": "TensorRT 10.7 INT8(EQ)+FP16",
}


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def row_from_pytorch():
    latency = load_json(require(ROOT / "artifacts/evaluation/pytorch_fp32/latency_metrics.json"))
    planning = latency["planning"]
    return {
        "label": "PyTorch 1.12 FP32",
        "model_mean_ms": latency["model_forward"]["mean_ms"],
        "model_p50_ms": latency["model_forward"]["p50_ms"],
        "fps_from_mean": latency["model_forward"]["fps_from_mean"],
        "e2e_mean_ms": latency["end_to_end"]["mean_ms"],
        "e2e_p50_ms": latency["end_to_end"]["p50_ms"],
        "avg_l2_m": planning["avg_l2_m"],
        "avg_col_percent": planning["avg_box_collision_percent"],
        "avg_point_col_percent": planning["avg_point_collision_percent"],
        "planning_reference_avg_l2_m": 0.0,
        "frames": latency["frames_total"],
    }


def row_from_trt(precision, label):
    result_dir = ROOT / f"artifacts/evaluation/tensorrt_{precision}"
    latency = load_json(require(result_dir / "latency_metrics.json"))
    planning = load_json(require(result_dir / "planning_metrics.json"))
    return {
        "label": label,
        "model_mean_ms": latency["model_enqueue"]["mean_ms"],
        "model_p50_ms": latency["model_enqueue"]["p50_ms"],
        "fps_from_mean": latency["model_enqueue"]["fps_from_mean"],
        "e2e_mean_ms": latency["end_to_end"]["mean_ms"],
        "e2e_p50_ms": latency["end_to_end"]["p50_ms"],
        "avg_l2_m": planning["avg_l2_m"],
        "avg_col_percent": planning["avg_box_collision_percent"],
        "avg_point_col_percent": planning["avg_point_collision_percent"],
        "planning_reference_avg_l2_m": planning["planning_reference_avg_l2_m"],
        "planning_reference_coordinate_mse": planning["planning_reference_coordinate_mse"],
        "frames": planning["frames"],
        "gpu": latency["gpu"],
        "tensorrt": latency["tensorrt_compile_version"],
    }


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def main():
    local = {
        "pytorch_fp32": row_from_pytorch(),
        "tensorrt_fp32": row_from_trt("fp32", "TensorRT 10.9 FP32"),
        "tensorrt_fp16": row_from_trt("fp16", "TensorRT 10.9 FP16"),
        "tensorrt_int8_eq_fp16": row_from_trt("int8_eq_fp16", "TensorRT 10.9 INT8(EQ)+FP16"),
    }

    comparisons = {}
    for key, row in local.items():
        reference = OFFICIAL[key]
        planning_tolerance = 1e-5 if key == "tensorrt_fp32" else 0.01
        comparisons[key] = {
            "avg_l2_abs_delta": abs(row["avg_l2_m"] - reference["l2"]),
            "avg_col_abs_delta_percentage_points": abs(row["avg_col_percent"] - reference["col"]),
            "planning_reference_abs_delta": abs(
                row["planning_reference_avg_l2_m"] - reference["planning"]),
            "avg_l2_close": abs(row["avg_l2_m"] - reference["l2"]) <= 0.02,
            "avg_col_close": abs(row["avg_col_percent"] - reference["col"]) <= 0.10,
            "planning_reference_close": abs(
                row["planning_reference_avg_l2_m"] - reference["planning"]) <= planning_tolerance,
        }
        comparisons[key]["accuracy_close"] = all(
            comparisons[key][name]
            for name in ("avg_l2_close", "avg_col_close", "planning_reference_close")
        )

    trt_p50 = [
        local["tensorrt_fp32"]["model_p50_ms"],
        local["tensorrt_fp16"]["model_p50_ms"],
        local["tensorrt_int8_eq_fp16"]["model_p50_ms"],
    ]
    latency_trend_matches = trt_p50[0] > trt_p50[1] > trt_p50[2]

    artifact_paths = [
        ROOT / "artifacts/checkpoints/bevformer_tiny_epoch_24_updated2.pth",
        ROOT / "artifacts/stage1/epoch_6.pth",
        ROOT / "artifacts/stage2/epoch_20.pth",
        ROOT / "artifacts/onnx/uniad_tiny_imgx0.25_cp.repaired.onnx",
        ROOT / "artifacts/onnx/uniad_tiny_int8_eq_dq_only.onnx",
        ROOT / "artifacts/onnx/quantization_inspection.json",
        ROOT / "artifacts/calibration/calib_data_shape0_901.npz",
        ROOT / "artifacts/engines/uniad_tiny_fp32.engine",
        ROOT / "artifacts/engines/uniad_tiny_fp16.engine",
        ROOT / "artifacts/engines/uniad_tiny_int8_eq_fp16.engine",
        ROOT / "UniAD_deploy/plugins/lib/lib_uniad_plugins_trt10.9_x86_cu118.so",
        ROOT.parent / "runtime/inference_app/enqueueV3/build/libuniad_plugin.so",
        ROOT.parent / "runtime/inference_app/enqueueV3/build/uniad",
    ]
    artifacts = {}
    for path in artifact_paths:
        require(path)
        try:
            artifact_key = path.relative_to(ROOT)
        except ValueError:
            artifact_key = path.relative_to(ROOT.parent)
        artifacts[str(artifact_key)] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    result = {
        "schema_version": 1,
        "source_revisions": {
            "DL4AGX": "9f7b29104c253d5bc68334e7b83b3eecb72d4572",
            "UniAD": "02fa68c5cbbd55261b34ad2c55c4c50fa6c120e6",
            "BEVFormer_TensorRT": "303d3140c14016047c07f9db73312af364f0dd7c",
        },
        "environment": {
            "gpu": local["tensorrt_fp32"]["gpu"],
            "tensorrt": local["tensorrt_fp32"]["tensorrt"],
            "pytorch": "1.12.1+cu116",
            "export_cuda_toolkit": "11.8",
            "modelopt": "0.29.0",
            "onnx": "1.17.0",
            "onnxruntime_gpu": "1.21.0",
            "cpp_build_conda_environment": "modelopt (unchanged)",
            "quantization_conda_environment": "modelopt_uniad_dl4agx (isolated clone)",
        },
        "official_reference": OFFICIAL,
        "local_results": local,
        "comparisons": comparisons,
        "latency_trend_fp32_gt_fp16_gt_int8_eq_fp16": latency_trend_matches,
        "artifacts": artifacts,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "comparison.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        "# UniAD-tiny quantization deployment report",
        "",
        "## Local environment",
        "",
        f"GPU: {result['environment']['gpu']}; TensorRT {result['environment']['tensorrt']}; "
        "PyTorch 1.12.1; ModelOpt 0.29.0; ONNX Runtime GPU 1.21.0.",
        "",
        "## Official reference (DRIVE Orin-X)",
        "",
        "| Precision | DL model latency (ms) | FPS | avg. L2 (m) | avg. Col (%) | planning reference L2 (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in OFFICIAL:
        value = OFFICIAL[key]
        lines.append(
            f"| {OFFICIAL_LABELS[key]} | {value['model_ms']:.4f} | {value['fps']:.2f} | "
            f"{value['l2']:.4f} | {value['col']:.2f} | {value['planning']:.7g} |"
        )

    lines += [
        "",
        "## Local measured results",
        "",
        "| Precision | Model mean (ms) | Model p50 (ms) | FPS (mean) | E2E mean (ms) | E2E p50 (ms) | avg. L2 (m) | avg. Col (%) | planning reference L2 (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in local.items():
        lines.append(
            f"| {value['label']} | {fmt(value['model_mean_ms'])} | {fmt(value['model_p50_ms'])} | "
            f"{fmt(value['fps_from_mean'], 2)} | {fmt(value['e2e_mean_ms'])} | "
            f"{fmt(value['e2e_p50_ms'])} | {fmt(value['avg_l2_m'])} | "
            f"{fmt(value['avg_col_percent'], 4)} | {value['planning_reference_avg_l2_m']:.7g} |"
        )

    lines += [
        "",
        "## Consistency checks",
        "",
        f"Latency precision trend (FP32 > FP16 > INT8(EQ)+FP16): **{'PASS' if latency_trend_matches else 'FAIL'}**.",
        "",
    ]
    for key, check in comparisons.items():
        lines.append(
            f"- {local[key]['label']}: **{'PASS' if check['accuracy_close'] else 'FAIL'}**; "
            f"L2 delta {check['avg_l2_abs_delta']:.6f} m, collision delta "
            f"{check['avg_col_abs_delta_percentage_points']:.6f} percentage points, planning-reference delta "
            f"{check['planning_reference_abs_delta']:.7g} m."
        )

    lines += [
        "",
        "PyTorch accuracy uses all 6019 validation frames. TensorRT accuracy and cross-framework planning comparison use the first 6018 frames, matching NVIDIA's documented C++ metadata range.",
        "",
        "`avg. Col` is the UniAD ego-box collision rate averaged over six future timesteps and reported as a percentage. Visualization and result-file writes are excluded from E2E latency.",
        "",
        "## Artifact hashes",
        "",
        "| Artifact | Bytes | SHA256 |",
        "|---|---:|---|",
    ]
    for name, artifact in artifacts.items():
        lines.append(f"| `{name}` | {artifact['size_bytes']} | `{artifact['sha256']}` |")
    lines += [
        "",
        "Official source: https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/uniad-trt",
        "",
    ]
    (REPORT_DIR / "final_report.md").write_text("\n".join(lines))
    print(REPORT_DIR / "final_report.md")


if __name__ == "__main__":
    main()
