#!/usr/bin/env python3
"""Compare two fixed-shape float32 tensor dumps from the UniAD runtime."""

import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_dump(path):
    manifest_path = path + ".manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("dtype") != "float32":
        raise ValueError("unsupported audit dtype: %s" % manifest.get("dtype"))
    frames = int(manifest["frames"])
    shape = tuple(int(value) for value in manifest["shape_per_frame"])
    expected_bytes = frames * int(np.prod(shape)) * np.dtype(np.float32).itemsize
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            "%s has %d bytes, expected %d" % (path, actual_bytes, expected_bytes)
        )
    values = np.memmap(
        path, dtype=np.float32, mode="r", shape=(frames,) + shape
    )
    return manifest, values


def main():
    args = parse_args()
    candidate_manifest, candidate = load_dump(args.candidate)
    reference_manifest, reference = load_dump(args.reference)
    for field in ("frames", "shape_per_frame", "temporal_protocol"):
        if candidate_manifest.get(field) != reference_manifest.get(field):
            raise ValueError("manifest mismatch for %s" % field)

    frames = int(candidate_manifest["frames"])
    difference = candidate - reference
    reduce_axes = tuple(range(1, difference.ndim))
    frame_mae = np.mean(np.abs(difference), axis=reduce_axes)
    frame_rmse = np.sqrt(np.mean(np.square(difference), axis=reduce_axes))
    reference_rms = np.sqrt(np.mean(np.square(reference), axis=reduce_axes))
    candidate_flat = candidate.reshape(frames, -1)
    reference_flat = reference.reshape(frames, -1)
    cosine = np.sum(candidate_flat * reference_flat, axis=1) / (
        np.linalg.norm(candidate_flat, axis=1)
        * np.linalg.norm(reference_flat, axis=1)
    )
    finite_relative = np.divide(
        frame_rmse,
        reference_rms,
        out=np.full_like(frame_rmse, np.nan),
        where=reference_rms != 0,
    )
    report = {
        "schema_version": 1,
        "candidate": os.path.realpath(args.candidate),
        "reference": os.path.realpath(args.reference),
        "frames": frames,
        "shape_per_frame": candidate_manifest["shape_per_frame"],
        "temporal_protocol": candidate_manifest.get("temporal_protocol"),
        "mae": {
            "mean": float(np.mean(frame_mae)),
            "p50": float(np.percentile(frame_mae, 50)),
            "p99": float(np.percentile(frame_mae, 99)),
            "max": float(np.max(frame_mae)),
        },
        "rmse_mean": float(np.mean(frame_rmse)),
        "relative_rmse_mean": float(np.nanmean(finite_relative)),
        "cosine": {
            "mean": float(np.mean(cosine)),
            "min": float(np.min(cosine)),
        },
        "frame_mae": [float(value) for value in frame_mae],
    }
    output_parent = os.path.dirname(os.path.abspath(args.output))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
