#!/usr/bin/env python3

import importlib
import json
import os

import numpy as np

from fixed_calibration_provider import FixedCalibrationDataProvider


def install_entropy_histogram_probe():
    report_path = os.environ.get("UNIAD_ENTROPY_HISTOGRAM_REPORT")
    if not report_path:
        return

    calibrate = importlib.import_module("onnxruntime.quantization.calibrate")

    def report_and_stop(collector):
        tensors = []
        for name, histogram in collector.histogram_dict.items():
            hist = histogram[0]
            edges = histogram[1]
            tensors.append({
                "name": name,
                "bins": int(hist.size),
                "nonzero_bins": int(np.count_nonzero(hist)),
                "edge_min": float(edges[0]),
                "edge_max": float(edges[-1]),
                "observed_min": float(histogram[2]),
                "observed_max": float(histogram[3]),
                "threshold": float(histogram[4]),
                "finite": bool(
                    np.isfinite(edges).all()
                    and np.isfinite(histogram[2:5]).all()
                ),
            })
        tensors.sort(key=lambda item: item["bins"], reverse=True)
        report = {
            "schema_version": 1,
            "tensor_count": len(tensors),
            "maximum_bins": max((item["bins"] for item in tensors), default=0),
            "nonfinite_tensor_count": sum(not item["finite"] for item in tensors),
            "tensors": tensors,
        }
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        raise RuntimeError(
            "Entropy histogram probe completed intentionally; "
            f"report={report_path}"
        )

    calibrate.HistogramCollector.compute_entropy = report_and_stop


def install_bounded_entropy_histograms():
    max_bins = int(os.environ.get("UNIAD_ENTROPY_MAX_BINS", "0"))
    if max_bins <= 0:
        return
    if max_bins < 256 or max_bins % 2:
        raise ValueError("UNIAD_ENTROPY_MAX_BINS must be an even integer >= 256")

    calibrate = importlib.import_module("onnxruntime.quantization.calibrate")
    original_compute_entropy = calibrate.HistogramCollector.compute_entropy
    report_path = os.environ.get("UNIAD_ENTROPY_REBIN_REPORT")

    def bounded_compute_entropy(collector):
        rebin_records = []
        for name, histogram in list(collector.histogram_dict.items()):
            hist = histogram[0]
            edges = histogram[1]
            if hist.size <= max_bins:
                continue

            group_size = int(np.ceil(hist.size / float(max_bins)))
            output_bins = int(np.ceil(hist.size / float(group_size)))
            if output_bins % 2:
                output_bins += 1
            if output_bins > max_bins:
                raise RuntimeError(
                    f"Unable to bound {name}: {hist.size} -> {output_bins} bins"
                )
            padded_bins = output_bins * group_size
            padding = padded_bins - hist.size
            left_padding = padding // 2
            right_padding = padding - left_padding
            if left_padding != right_padding:
                raise RuntimeError(
                    f"Asymmetric histogram padding for {name}: {padding} bins"
                )

            rebinned = np.pad(hist, (left_padding, right_padding)).reshape(
                output_bins, group_size
            ).sum(axis=1)
            bin_width = edges[1] - edges[0]
            lower = edges[0] - left_padding * bin_width
            upper = edges[-1] + right_padding * bin_width
            threshold = max(abs(float(lower)), abs(float(upper)))
            rebinned_edges = np.linspace(
                -threshold, threshold, output_bins + 1, dtype=edges.dtype
            )
            if int(rebinned.sum()) != int(hist.sum()):
                raise RuntimeError(f"Histogram mass changed while bounding {name}")
            if not np.allclose(
                    rebinned_edges[0], -rebinned_edges[-1], rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"Histogram symmetry changed while bounding {name}")

            collector.histogram_dict[name] = (
                rebinned,
                rebinned_edges,
                histogram[2],
                histogram[3],
                np.array(rebinned_edges[-1], dtype=edges.dtype),
            )
            rebin_records.append({
                "name": name,
                "input_bins": int(hist.size),
                "output_bins": int(rebinned.size),
                "group_size": group_size,
                "symmetric_zero_padding_per_side": left_padding,
                "input_count": int(hist.sum()),
                "output_count": int(rebinned.sum()),
                "input_edge_min_max": [float(edges[0]), float(edges[-1])],
                "output_edge_min_max": [
                    float(rebinned_edges[0]), float(rebinned_edges[-1])
                ],
            })

        rebin_records.sort(key=lambda item: item["input_bins"], reverse=True)
        report = {
            "schema_version": 1,
            "method": (
                "symmetric zero padding followed by exact contiguous-bin summation; "
                "ORT entropy/KL threshold search is unchanged"
            ),
            "maximum_bins": max_bins,
            "rebinned_tensor_count": len(rebin_records),
            "mass_preserved": all(
                item["input_count"] == item["output_count"]
                for item in rebin_records
            ),
            "tensors": rebin_records,
        }
        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print(
            "UniAD bounded entropy histograms: "
            f"rebinned={len(rebin_records)}, max_bins={max_bins}",
            flush=True,
        )
        return original_compute_entropy(collector)

    calibrate.HistogramCollector.compute_entropy = bounded_compute_entropy


def main():
    calib_utils = importlib.import_module("modelopt.onnx.quantization.calib_utils")
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    cli_module = importlib.import_module("modelopt.onnx.quantization.__main__")

    # Patch only the provider symbol used by quantize(); the environment and
    # installed ModelOpt source remain untouched.
    calib_utils.CalibrationDataProvider = FixedCalibrationDataProvider
    quantize_module.CalibrationDataProvider = FixedCalibrationDataProvider
    install_entropy_histogram_probe()
    install_bounded_entropy_histograms()
    cli_module.main()


if __name__ == "__main__":
    main()
