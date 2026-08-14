#!/usr/bin/env python3

"""Compare packed TensorRT occupancy with a PyTorch forward audit."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-packbits", required=True, type=Path)
    parser.add_argument("--pytorch-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    with np.load(args.pytorch_audit) as audit:
        reference = audit["seg_out_packed"].astype(np.uint8, copy=False)
        shape = tuple(int(value) for value in audit["seg_out_shape"])
    frame_count, packed_bytes = reference.shape
    actual = np.fromfile(args.trt_packbits, dtype=np.uint8)
    expected_bytes = frame_count * packed_bytes
    if actual.size != expected_bytes:
        raise ValueError(
            f"TensorRT dump has {actual.size} bytes, expected {expected_bytes}")
    actual = actual.reshape(frame_count, packed_bytes)
    bit_count = int(np.prod(shape, dtype=np.int64))
    actual_bits = np.unpackbits(actual, axis=1, bitorder="little")[:, :bit_count]
    reference_bits = np.unpackbits(
        reference, axis=1, bitorder="little")[:, :bit_count]
    actual_bits = actual_bits.astype(np.bool_, copy=False).reshape(
        (frame_count,) + shape)
    reference_bits = reference_bits.astype(np.bool_, copy=False).reshape(
        (frame_count,) + shape)

    intersection = np.logical_and(actual_bits, reference_bits)
    union = np.logical_or(actual_bits, reference_bits)
    actual_positive = int(actual_bits.sum())
    reference_positive = int(reference_bits.sum())
    intersection_count = int(intersection.sum())
    union_count = int(union.sum())
    horizon_axes = tuple(index for index in range(actual_bits.ndim) if index != 2)
    horizon_intersection = intersection.sum(axis=horizon_axes)
    horizon_union = union.sum(axis=horizon_axes)
    horizon_actual = actual_bits.sum(axis=horizon_axes)
    horizon_reference = reference_bits.sum(axis=horizon_axes)

    result = {
        "schema_version": 1,
        "frames": frame_count,
        "seg_out_shape": list(shape),
        "pytorch_positive_cells": reference_positive,
        "tensorrt_positive_cells": actual_positive,
        "intersection_cells": intersection_count,
        "union_cells": union_count,
        "iou_percent": 100.0 * intersection_count / max(union_count, 1),
        "precision_percent": 100.0 * intersection_count / max(actual_positive, 1),
        "recall_percent": 100.0 * intersection_count / max(reference_positive, 1),
        "byte_exact_frames": int(np.all(actual == reference, axis=1).sum()),
        "iou_by_horizon_percent": (
            100.0 * horizon_intersection / np.maximum(horizon_union, 1)
        ).tolist(),
        "pytorch_positive_by_horizon": horizon_reference.tolist(),
        "tensorrt_positive_by_horizon": horizon_actual.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
