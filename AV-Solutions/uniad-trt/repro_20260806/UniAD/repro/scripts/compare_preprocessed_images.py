#!/usr/bin/env python3

"""Compare C++ preprocessed images with saved PyTorch DataLoader tensors."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-images", required=True, type=Path)
    parser.add_argument("--pytorch-images", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    references = [
        np.load(args.pytorch_images / f"{frame}.npy").astype(
            np.float32, copy=False)
        for frame in range(args.frames)
    ]
    shape = references[0].shape
    if any(reference.shape != shape for reference in references):
        raise ValueError("PyTorch image shapes are not constant")
    elements_per_frame = int(np.prod(shape, dtype=np.int64))
    actual = np.fromfile(args.trt_images, dtype=np.float32)
    if actual.size != args.frames * elements_per_frame:
        raise ValueError(
            f"C++ dump has {actual.size} values, expected "
            f"{args.frames * elements_per_frame}")
    actual = actual.reshape((args.frames,) + shape)
    reference = np.stack(references)
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    per_frame_max_abs = np.max(np.abs(delta), axis=tuple(range(1, delta.ndim)))
    per_frame_mae = np.mean(np.abs(delta), axis=tuple(range(1, delta.ndim)))
    result = {
        "schema_version": 1,
        "frames": args.frames,
        "shape_per_frame": list(shape),
        "byte_exact_frames": int(np.all(actual == reference, axis=tuple(
            range(1, actual.ndim))).sum()),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "root_mean_square": float(np.sqrt(np.mean(np.square(delta)))),
        "per_frame_max_abs": per_frame_max_abs.tolist(),
        "per_frame_mean_abs": per_frame_mae.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
