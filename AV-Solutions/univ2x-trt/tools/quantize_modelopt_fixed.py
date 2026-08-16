import argparse
import hashlib
import importlib
import json
import os
import platform
import time

import modelopt
import numpy as np
import onnx
import onnxruntime
import tensorrt
from modelopt.onnx.utils import get_input_names, get_input_shapes, parse_shapes_spec


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_data_artifacts(onnx_path):
    model = onnx.load(onnx_path, load_external_data=False)
    locations = set()
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    artifacts = []
    for location in sorted(locations):
        path = os.path.join(os.path.dirname(onnx_path), location)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        artifacts.append({
            "path": os.path.abspath(path),
            "bytes": os.path.getsize(path),
            "sha256": sha256(path),
        })
    return artifacts


class FixedCalibrationDataProvider:
    """ModelOpt 0.29 provider with one independent dictionary per iteration."""

    def __init__(self, onnx_path, calibration_data, calibration_shapes=None):
        model = onnx.load(onnx_path)
        input_names = get_input_names(model)
        input_shapes = (
            {} if calibration_shapes is None
            else parse_shapes_spec(calibration_shapes)
        )
        inferred_shapes = get_input_shapes(model)
        for name in input_names:
            input_shapes.setdefault(name, inferred_shapes[name])
        if set(input_names) != set(calibration_data):
            raise ValueError({
                "missing": sorted(set(input_names) - set(calibration_data)),
                "extra": sorted(set(calibration_data) - set(input_names)),
            })

        iteration_counts = {}
        for name in input_names:
            per_iteration = int(input_shapes[name][0])
            stored = int(calibration_data[name].shape[0])
            if per_iteration <= 0 or stored % per_iteration:
                raise ValueError(
                    "%s stored first dimension %d is incompatible with %d"
                    % (name, stored, per_iteration)
                )
            iteration_counts[name] = stored // per_iteration
        if len(set(iteration_counts.values())) != 1:
            raise ValueError("Inconsistent calibration iterations: %s" % iteration_counts)
        iteration_count = next(iter(iteration_counts.values()))

        # ModelOpt 0.29.0 uses `[{}] * iteration_count` here, aliasing every
        # entry and silently repeating only the final calibration sample.
        self.calibration_data_list = [dict() for _ in range(iteration_count)]
        for name in input_names:
            chunks = np.split(calibration_data[name], iteration_count, axis=0)
            for index, chunk in enumerate(chunks):
                self.calibration_data_list[index][name] = chunk
        self.calibration_data_reader = iter(self.calibration_data_list)

    def get_next(self):
        return next(self.calibration_data_reader, None)

    def get_first(self):
        if not self.calibration_data_list:
            raise ValueError("Calibration data is empty")
        return self.calibration_data_list[0]


class ManifestCalibrationDataProvider:
    """Lazy ModelOpt reader for per-frame, variable-shape NPZ samples."""

    target_track_count = None
    target_coop_count = None

    def __init__(self, onnx_path, calibration_data, calibration_shapes=None):
        del calibration_shapes
        if set(calibration_data) != {"__manifest__"}:
            raise ValueError("Manifest provider did not receive a manifest path")
        self.manifest_path = os.path.abspath(calibration_data["__manifest__"])
        with open(self.manifest_path) as handle:
            self.manifest = json.load(handle)
        model = onnx.load(onnx_path, load_external_data=False)
        self.input_names = get_input_names(model)
        self.directory = os.path.dirname(self.manifest_path)
        self.sample_paths = [
            os.path.join(self.directory, entry["file"])
            for entry in self.manifest["samples"]
        ]
        if not self.sample_paths:
            raise ValueError("Calibration manifest is empty")
        expected_names = set(self.input_names)
        for entry, path in zip(self.manifest["samples"], self.sample_paths):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            if set(entry["shapes"]) != expected_names:
                raise ValueError({
                    "file": path,
                    "missing": sorted(expected_names - set(entry["shapes"])),
                    "extra": sorted(set(entry["shapes"]) - expected_names),
                })
        self.rewind()

    @staticmethod
    def _resize_first_dimension(name, value, target):
        if target is None or value.shape[0] == target:
            return value
        if value.shape[0] > target:
            return value[:target].copy()
        fill_value = 0
        if (name.endswith("intances3")
                or name.endswith("intances4")
                or name == "coop_match_vehicle_index"):
            fill_value = -1
        elif name.endswith("intances12"):
            fill_value = 1
        padding = np.full(
            (target - value.shape[0],) + value.shape[1:],
            fill_value,
            dtype=value.dtype,
        )
        return np.concatenate([value, padding], axis=0)

    def _load(self, path):
        with np.load(path) as archive:
            sample = {name: archive[name] for name in self.input_names}
        for name, value in sample.items():
            if name.startswith("prev_track_intances"):
                sample[name] = self._resize_first_dimension(
                    name, value, self.target_track_count
                )
            elif (name.startswith("coop_track_intances")
                    or name == "coop_match_vehicle_index"):
                sample[name] = self._resize_first_dimension(
                    name, value, self.target_coop_count
                )
        return sample

    def get_next(self):
        if self._index >= len(self.sample_paths):
            return None
        sample = self._load(self.sample_paths[self._index])
        self._index += 1
        return sample

    def get_first(self):
        return self._load(self.sample_paths[0])

    def rewind(self):
        self._index = 0


def provider_preflight(onnx_path, calibration_data, calibration_shapes):
    provider = FixedCalibrationDataProvider(
        onnx_path, calibration_data, calibration_shapes
    )
    iterations = provider.calibration_data_list
    return {
        "iterations": len(iterations),
        "independent_dictionary_ids": len({id(item) for item in iterations}),
        "use_prev_bev": [
            int(item["use_prev_bev"].reshape(-1)[0]) for item in iterations
        ],
        "commands": (
            [int(item["command"].reshape(-1)[0]) for item in iterations]
            if "command" in iterations[0] else None
        ),
        "input_shapes": {
            name: list(value.shape) for name, value in iterations[0].items()
        },
    }


def manifest_provider_preflight(onnx_path, manifest_path):
    provider = ManifestCalibrationDataProvider(
        onnx_path, {"__manifest__": manifest_path}
    )
    entries = provider.manifest["samples"]
    first = provider.get_first()
    return {
        "iterations": len(entries),
        "dataset_split": provider.manifest.get("dataset_split"),
        "use_prev_bev": [entry["use_prev_bev"] for entry in entries],
        "commands": [entry.get("command") for entry in entries],
        "track_counts": [entry["track_count"] for entry in entries],
        "coop_track_counts": [entry.get("coop_track_count") for entry in entries],
        "normalized_track_count": provider.target_track_count,
        "normalized_coop_track_count": provider.target_coop_count,
        "first_input_shapes": {
            name: list(value.shape) for name, value in first.items()
        },
        "total_bytes": int(provider.manifest.get("total_bytes", 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    calibration_group = parser.add_mutually_exclusive_group(required=True)
    calibration_group.add_argument("--calibration")
    calibration_group.add_argument("--calibration-manifest")
    parser.add_argument("--calibration-report")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--use-external-data-format", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--manifest-track-count", type=int)
    parser.add_argument("--manifest-coop-count", type=int)
    args = parser.parse_args()

    paths = [args.onnx, args.plugin]
    paths.append(args.calibration or args.calibration_manifest)
    if args.calibration_report:
        paths.append(args.calibration_report)
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    calibration_report = None
    calibration_shapes = None
    if args.calibration:
        if not args.calibration_report:
            raise ValueError("--calibration requires --calibration-report")
        with open(args.calibration_report) as handle:
            calibration_report = json.load(handle)
        calibration_shapes = calibration_report["output"]["calibration_shapes"]
        with np.load(args.calibration) as archive:
            calibration_data = {name: archive[name] for name in archive.files}
        preflight = provider_preflight(
            args.onnx, calibration_data, calibration_shapes
        )
        expected_prev = calibration_report["output"]["use_prev_bev"]
        expected_commands = calibration_report["output"]["commands"]
        if preflight["use_prev_bev"] != expected_prev:
            raise ValueError((preflight["use_prev_bev"], expected_prev))
        if preflight["commands"] != expected_commands:
            raise ValueError((preflight["commands"], expected_commands))
        if preflight["independent_dictionary_ids"] != preflight["iterations"]:
            raise ValueError("Calibration iteration dictionaries are still aliased")
        provider_class = FixedCalibrationDataProvider
    else:
        if args.manifest_track_count is not None and args.manifest_track_count <= 0:
            raise ValueError("--manifest-track-count must be positive")
        if args.manifest_coop_count is not None and args.manifest_coop_count <= 0:
            raise ValueError("--manifest-coop-count must be positive")
        ManifestCalibrationDataProvider.target_track_count = (
            args.manifest_track_count
        )
        ManifestCalibrationDataProvider.target_coop_count = (
            args.manifest_coop_count
        )
        calibration_data = {"__manifest__": os.path.abspath(args.calibration_manifest)}
        preflight = manifest_provider_preflight(args.onnx, args.calibration_manifest)
        if preflight["dataset_split"] != "train":
            raise ValueError("Formal calibration manifest must use the train split")
        provider_class = ManifestCalibrationDataProvider

    quantize_module = importlib.import_module(
        "modelopt.onnx.quantization.quantize"
    )
    quantize_module.CalibrationDataProvider = provider_class
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    start = time.perf_counter()
    quantize_module.quantize(
        os.path.abspath(args.onnx),
        quantize_mode="int8",
        calibration_data=calibration_data,
        calibration_method="entropy",
        calibration_shapes=calibration_shapes,
        calibration_eps=["trt", "cuda:0", "cpu"],
        op_types_to_exclude=["MatMul"],
        use_external_data_format=args.use_external_data_format,
        keep_intermediate_files=False,
        output_path=output_path,
        verbose=args.verbose,
        trt_plugins=os.path.abspath(args.plugin),
        high_precision_dtype="fp32",
        mha_accumulation_dtype=None,
        disable_mha_qdq=False,
        dq_only=True,
        passes=None,
        simplify=args.simplify,
    )
    elapsed = time.perf_counter() - start
    if not os.path.isfile(output_path):
        raise RuntimeError("ModelOpt did not produce %s" % output_path)

    report = {
        "modelopt_version": modelopt.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "tensorrt_version": tensorrt.__version__,
        "python_version": platform.python_version(),
        "runtime_environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
            "path": os.environ.get("PATH"),
        },
        "provider_workaround": {
            "applied": True,
            "provider": provider_class.__name__,
            "reason": (
                "ModelOpt 0.29.0 aliases fixed-shape entries; the manifest "
                "reader also supports lazy variable-shape calibration"
            ),
            "installed_source_unchanged": True,
        },
        "preflight": preflight,
        "configuration": {
            "quantize_mode": "int8",
            "calibration_method": "entropy",
            "calibration_eps": ["trt", "cuda:0", "cpu"],
            "calibration_shapes": calibration_shapes,
            "op_types_to_exclude": ["MatMul"],
            "high_precision_dtype": "fp32",
            "dq_only": True,
            "simplify": args.simplify,
            "use_external_data_format": args.use_external_data_format,
            "manifest_track_count": args.manifest_track_count,
            "manifest_coop_count": args.manifest_coop_count,
        },
        "inputs": {
            "onnx": os.path.abspath(args.onnx),
            "onnx_sha256": sha256(args.onnx),
            "calibration": os.path.abspath(
                args.calibration or args.calibration_manifest
            ),
            "calibration_sha256": sha256(
                args.calibration or args.calibration_manifest
            ),
            "plugin": os.path.abspath(args.plugin),
            "plugin_sha256": sha256(args.plugin),
        },
        "output": {
            "onnx": output_path,
            "bytes": os.path.getsize(output_path),
            "sha256": sha256(output_path),
            "external_data": external_data_artifacts(output_path),
        },
        "elapsed_seconds": elapsed,
    }
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
