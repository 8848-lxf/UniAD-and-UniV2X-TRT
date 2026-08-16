#!/usr/bin/env python3

import importlib
import gc
import json
import os
import types

import numpy as np

from fixed_calibration_provider import FixedCalibrationDataProvider


def _tensor_components(tensor_names, group_qdq_tensors):
    parent = {name: name for name in tensor_names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for current, group in (group_qdq_tensors or {}).items():
        members = [name for name in [current, *group] if name in parent]
        for member in members[1:]:
            union(members[0], member)

    components = {}
    for name in sorted(tensor_names):
        components.setdefault(find(name), []).append(name)
    return sorted(components.values(), key=lambda item: (-len(item), item[0]))


def _chunk_tensor_components(components, maximum_tensors):
    chunks = []
    current = []
    for component in components:
        if current and len(current) + len(component) > maximum_tensors:
            chunks.append(current)
            current = []
        current.extend(component)
    if current:
        chunks.append(current)
    return chunks


def install_chunked_entropy_calibrator():
    maximum_tensors = int(os.environ.get("UNIAD_ENTROPY_TENSORS_PER_CHUNK", "0"))
    if maximum_tensors <= 0:
        return

    calibrate = importlib.import_module("onnxruntime.quantization.calibrate")
    ort_patching = importlib.import_module("modelopt.onnx.quantization.ort_patching")
    original_create_calibrator = calibrate.create_calibrator
    report_path = os.environ.get("UNIAD_ENTROPY_CHUNK_REPORT")

    class ChunkedEntropyCalibrator:
        def __init__(
            self,
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            extra_options,
        ):
            self.model_path = model
            self.op_types_to_calibrate = op_types_to_calibrate
            self.augmented_model_path = augmented_model_path
            self.use_external_data_format = use_external_data_format
            self.extra_options = dict(extra_options)
            self.ranges = None

            template = calibrate.EntropyCalibrater(
                model,
                op_types_to_calibrate,
                augmented_model_path,
                use_external_data_format=use_external_data_format,
                symmetric=extra_options.get("symmetric", False),
                num_bins=extra_options.get("num_bins", 128),
                num_quantized_bins=extra_options.get("num_quantized_bins", 128),
            )
            tensors, value_infos = template.select_tensors_to_calibrate(template.model)
            self.value_infos = value_infos
            components = _tensor_components(
                tensors,
                extra_options.get("group_qdq_tensors"),
            )
            self.chunks = _chunk_tensor_components(components, maximum_tensors)
            del template
            gc.collect()

            report = {
                "schema_version": 1,
                "method": "entropy",
                "status": "planned",
                "tensor_count": len(tensors),
                "maximum_tensors_per_chunk": maximum_tensors,
                "chunk_count": len(self.chunks),
                "completed_chunks": [],
                "chunks": [
                    {
                        "index": index,
                        "tensor_count": len(chunk),
                        "tensors": chunk,
                    }
                    for index, chunk in enumerate(self.chunks)
                ],
                "group_qdq_components_kept_together": True,
            }
            self.report = report
            self.report_path = report_path
            self._write_report()

        def _write_report(self):
            if report_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.report_path)), exist_ok=True)
                with open(self.report_path, "w", encoding="utf-8") as handle:
                    json.dump(self.report, handle, indent=2, sort_keys=True)
                    handle.write("\n")

        def _announce_plan(self):
            print(
                "UniAD chunked entropy calibration: "
                f"tensors={self.report['tensor_count']}, chunks={len(self.chunks)}, "
                f"maximum_tensors_per_chunk={maximum_tensors}",
                flush=True,
            )

        def collect_data(self, data_reader):
            if not hasattr(data_reader, "rewind"):
                raise TypeError("Chunked entropy calibration requires a rewindable data reader")

            self._announce_plan()
            self.report["status"] = "running"
            self._write_report()
            merged_data = {}
            original_group_map = self.extra_options.get("group_qdq_tensors") or {}
            for index, chunk in enumerate(self.chunks):
                chunk_set = set(chunk)
                chunk_path = os.path.join(
                    os.path.dirname(self.augmented_model_path),
                    f"augmented_model.chunk_{index:03d}.onnx",
                )
                child = calibrate.EntropyCalibrater(
                    self.model_path,
                    self.op_types_to_calibrate,
                    chunk_path,
                    use_external_data_format=self.use_external_data_format,
                    symmetric=self.extra_options.get("symmetric", False),
                    num_bins=self.extra_options.get("num_bins", 128),
                    num_quantized_bins=self.extra_options.get("num_quantized_bins", 128),
                )

                def select_chunk_tensors(_child, _model):
                    return chunk_set, self.value_infos

                child.select_tensors_to_calibrate = types.MethodType(
                    select_chunk_tensors,
                    child,
                )
                child_options = dict(self.extra_options)
                child_options["group_qdq_tensors"] = {
                    current: [name for name in group if name in chunk_set]
                    for current, group in original_group_map.items()
                    if current in chunk_set
                }
                child.augment_graph()
                ort_patching._create_inference_session_with_ep_config(
                    child,
                    **child_options,
                )
                data_reader.rewind()
                print(
                    f"UniAD entropy chunk {index + 1}/{len(self.chunks)}: "
                    f"tensors={len(chunk)}",
                    flush=True,
                )
                child.collect_data(data_reader)
                chunk_ranges = child.compute_data()
                overlap = set(merged_data).intersection(chunk_ranges.data)
                if overlap:
                    raise RuntimeError(
                        f"Calibration tensor appeared in multiple chunks: {sorted(overlap)[:5]}"
                    )
                merged_data.update(chunk_ranges.data)
                child.infer_session = None
                del child, chunk_ranges
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
                gc.collect()
                self.report["completed_chunks"].append(index)
                self._write_report()

            self.ranges = calibrate.TensorsData(
                calibrate.CalibrationMethod.Entropy,
                merged_data,
            )
            self.report["status"] = "complete"
            self._write_report()

        def compute_data(self):
            if self.ranges is None:
                raise ValueError("No chunked entropy calibration data was collected")
            return self.ranges

    def create_calibrator(
        model,
        op_types_to_calibrate=None,
        augmented_model_path="augmented_model.onnx",
        calibrate_method=calibrate.CalibrationMethod.MinMax,
        use_external_data_format=False,
        extra_options=None,
    ):
        if calibrate_method != calibrate.CalibrationMethod.Entropy:
            return original_create_calibrator(
                model,
                op_types_to_calibrate,
                augmented_model_path,
                calibrate_method,
                use_external_data_format,
                extra_options or {},
            )
        return ChunkedEntropyCalibrator(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            extra_options or {},
        )

    create_calibrator._uniad_chunked_entropy = True
    calibrate.create_calibrator = create_calibrator


def install_chunked_entropy_after_modelopt_configure():
    maximum_tensors = int(os.environ.get("UNIAD_ENTROPY_TENSORS_PER_CHUNK", "0"))
    if maximum_tensors <= 0:
        return

    int8_module = importlib.import_module("modelopt.onnx.quantization.int8")
    original_configure_ort = int8_module.configure_ort

    def configure_ort_then_install(*args, **kwargs):
        result = original_configure_ort(*args, **kwargs)
        # configure_ort() calls ModelOpt's patch_ort_modules(), which restores its
        # monolithic calibrator factory. Install the project-local wrapper after it.
        install_chunked_entropy_calibrator()
        return result

    int8_module.configure_ort = configure_ort_then_install


def install_bounded_trt_calibration_session():
    workspace_gib = int(os.environ.get("UNIAD_TRT_CALIBRATION_WORKSPACE_GIB", "0"))
    if workspace_gib <= 0:
        return

    ort = importlib.import_module("onnxruntime")
    ort_patching = importlib.import_module("modelopt.onnx.quantization.ort_patching")

    def create_inference_session(calibrator, **kwargs):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = list(kwargs.get("execution_providers", []))
        plugin_paths = kwargs.get("trt_extra_plugin_lib_paths", None)

        if plugin_paths is not None:
            trt_options = {
                "trt_max_workspace_size": workspace_gib * 1024 * 1024 * 1024,
                "trt_auxiliary_streams": 0,
                "trt_builder_optimization_level": 0,
            }
            if plugin_paths:
                trt_options["trt_extra_plugin_lib_paths"] = plugin_paths
            providers = [
                provider for provider in providers
                if provider != "TensorrtExecutionProvider"
            ]
            providers.insert(0, ("TensorrtExecutionProvider", trt_options))

        calibrator.infer_session = ort.InferenceSession(
            calibrator.augmented_model_path,
            sess_options=session_options,
            providers=providers,
        )
        calibrator.group_qdq_tensors = kwargs.get("group_qdq_tensors", None)

    ort_patching._create_inference_session_with_ep_config = create_inference_session
    print(
        "UniAD bounded TensorRT calibration session: "
        f"workspace={workspace_gib} GiB, auxiliary_streams=0, builder_level=0",
        flush=True,
    )


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
    install_bounded_trt_calibration_session()
    install_chunked_entropy_after_modelopt_configure()
    install_entropy_histogram_probe()
    install_bounded_entropy_histograms()
    cli_module.main()


if __name__ == "__main__":
    main()
