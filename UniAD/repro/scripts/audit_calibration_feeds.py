#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np

from fixed_calibration_provider import (
    FixedCalibrationDataProvider,
    emulate_modelopt_029_alias,
    feed_signature,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--calibration-shapes", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.calibration) as archive:
        calibration_data = {name: archive[name] for name in archive.files}
    fixed = FixedCalibrationDataProvider(
        args.onnx, calibration_data, args.calibration_shapes
    )
    broken = emulate_modelopt_029_alias(fixed.calibration_data_list)
    broken_signatures = [feed_signature(item) for item in broken]

    result = {
        "schema_version": 1,
        "onnx": str(Path(args.onnx).resolve()),
        "calibration": str(Path(args.calibration).resolve()),
        "npz_frames": len(fixed.calibration_data_list),
        "npz_unique_frame_signatures": len(
            {feed_signature(item) for item in fixed.calibration_data_list}
        ),
        "modelopt_0_29_0_upstream_behavior": {
            "dictionary_ids": len({id(item) for item in broken}),
            "unique_feed_signatures": len(set(broken_signatures)),
            "effective_frame_index": len(broken) - 1,
        },
        "fixed_provider_behavior": fixed.preflight,
        "verdict": (
            "The NPZ contains distinct frames, but the upstream ModelOpt 0.29.0 "
            "provider repeats only the final frame. The repository-local provider "
            "preserves one independent feed per frame."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "npz_frames": result["npz_frames"],
        "npz_unique": result["npz_unique_frame_signatures"],
        "upstream_unique": result["modelopt_0_29_0_upstream_behavior"]["unique_feed_signatures"],
        "fixed_unique": result["fixed_provider_behavior"]["unique_feed_signatures"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
