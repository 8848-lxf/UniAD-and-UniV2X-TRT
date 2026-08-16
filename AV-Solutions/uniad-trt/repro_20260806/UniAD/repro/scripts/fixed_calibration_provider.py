#!/usr/bin/env python3

import hashlib
import json
import os

import numpy as np
import onnx
from modelopt.onnx.utils import get_input_names, get_input_shapes, parse_shapes_spec


def feed_signature(feed):
    digest = hashlib.sha256()
    for name in sorted(feed):
        value = np.ascontiguousarray(feed[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


class FixedCalibrationDataProvider:
    """ModelOpt 0.29 provider with an independent dictionary per frame."""

    def __init__(self, onnx_path, calibration_data, calibration_shapes=None):
        model = onnx.load(onnx_path, load_external_data=False)
        input_names = get_input_names(model)
        input_shapes = (
            {} if calibration_shapes is None else parse_shapes_spec(calibration_shapes)
        )
        inferred_shapes = get_input_shapes(model)
        for name in input_names:
            input_shapes.setdefault(name, inferred_shapes[name])

        if isinstance(calibration_data, np.ndarray):
            if len(input_names) != 1:
                raise ValueError("A tensor calibration package requires a one-input model")
            calibration_data = {input_names[0]: calibration_data}
        if not isinstance(calibration_data, dict):
            raise TypeError("Calibration data must be an ndarray or input-name dictionary")
        if set(input_names) != set(calibration_data):
            raise ValueError(
                {
                    "missing": sorted(set(input_names) - set(calibration_data)),
                    "extra": sorted(set(calibration_data) - set(input_names)),
                }
            )

        iteration_counts = {}
        for name in input_names:
            per_iteration = int(input_shapes[name][0])
            stored = int(calibration_data[name].shape[0])
            if per_iteration <= 0 or stored % per_iteration:
                raise ValueError(
                    f"{name}: stored first dimension {stored} is incompatible with "
                    f"per-frame dimension {per_iteration}"
                )
            iteration_counts[name] = stored // per_iteration
        if len(set(iteration_counts.values())) != 1:
            raise ValueError(f"Inconsistent calibration frame counts: {iteration_counts}")

        iteration_count = next(iter(iteration_counts.values()))
        if iteration_count <= 0:
            raise ValueError("Calibration package contains no frames")

        # ModelOpt 0.29.0 uses `[{}] * iteration_count` here. All entries then
        # alias one dictionary and are overwritten until only the final frame remains.
        self.calibration_data_list = [dict() for _ in range(iteration_count)]
        for name in input_names:
            for index, chunk in enumerate(
                np.split(calibration_data[name], iteration_count, axis=0)
            ):
                self.calibration_data_list[index][name] = chunk

        independent_ids = len({id(item) for item in self.calibration_data_list})
        signatures = [feed_signature(item) for item in self.calibration_data_list]
        unique_signatures = len(set(signatures))
        if independent_ids != iteration_count:
            raise RuntimeError("Calibration feed dictionaries are aliased")
        if unique_signatures != iteration_count:
            raise RuntimeError(
                f"Calibration feeds are not unique: {unique_signatures}/{iteration_count}"
            )

        self.calibration_data_reader = iter(self.calibration_data_list)
        self.preflight = {
            "frames": iteration_count,
            "independent_dictionary_ids": independent_ids,
            "unique_feed_signatures": unique_signatures,
            "feed_sha256": signatures,
            "input_shapes": {
                name: list(value.shape)
                for name, value in self.calibration_data_list[0].items()
            },
            "provider_workaround": {
                "applied": True,
                "reason": "ModelOpt 0.29.0 aliases fixed-shape calibration dictionaries",
                "installed_modelopt_source_unchanged": True,
            },
        }
        report_path = os.environ.get("UNIAD_CALIBRATION_PROVIDER_REPORT")
        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(self.preflight, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print(
            "UniAD calibration provider: "
            f"frames={iteration_count}, independent_dicts={independent_ids}, "
            f"unique_feeds={unique_signatures}",
            flush=True,
        )

    def get_next(self):
        return next(self.calibration_data_reader, None)

    def rewind(self):
        self.calibration_data_reader = iter(self.calibration_data_list)

    def get_first(self):
        if not self.calibration_data_list:
            raise ValueError("Calibration data is empty")
        return self.calibration_data_list[0]


def emulate_modelopt_029_alias(calibration_data_list):
    """Reproduce the upstream alias defect for an evidence-only comparison."""
    aliased = [{}] * len(calibration_data_list)
    for name in calibration_data_list[0]:
        for index, source in enumerate(calibration_data_list):
            aliased[index][name] = source[name]
    return aliased
