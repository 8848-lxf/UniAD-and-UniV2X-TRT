import argparse
import hashlib
import json
import os

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_samples(path, sample_count):
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    samples = [dict() for _ in range(sample_count)]
    for name, value in arrays.items():
        if value.shape[0] % sample_count:
            raise ValueError(
                "%s first dimension %d is not divisible by %d"
                % (name, value.shape[0], sample_count)
            )
        chunks = np.split(value, sample_count, axis=0)
        for index, chunk in enumerate(chunks):
            samples[index][name] = chunk
    return samples


def normalize_sample(sample, target_shapes):
    normalized = {}
    cropped = {}
    for name, target_shape in target_shapes.items():
        if name not in sample:
            raise KeyError("Missing input %s" % name)
        value = sample[name]
        if tuple(value.shape) == target_shape:
            normalized[name] = value.copy()
            continue
        if (
            value.ndim != len(target_shape)
            or tuple(value.shape[1:]) != target_shape[1:]
            or value.shape[0] < target_shape[0]
        ):
            raise ValueError(
                "Cannot normalize %s from %s to %s"
                % (name, tuple(value.shape), target_shape)
            )
        normalized[name] = value[: target_shape[0]].copy()
        cropped[name] = {
            "source_shape": list(value.shape),
            "target_shape": list(target_shape),
        }
    return normalized, cropped


def tensor_audit(arrays):
    nonfinite = {}
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating):
            count = int(np.size(value) - np.count_nonzero(np.isfinite(value)))
            if count:
                nonfinite[name] = count
    return nonfinite


def calibration_shapes(target_shapes):
    return ",".join(
        "%s:%s" % (name, "x".join(str(dim) for dim in shape))
        for name, shape in target_shapes.items()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-start", required=True)
    parser.add_argument("--scene-start-samples", type=int, required=True)
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--keep-scene-start", type=int, required=True)
    parser.add_argument("--temporal-copies", type=int, required=True)
    parser.add_argument("--temporal-commands", default="")
    args = parser.parse_args()

    base_samples = load_samples(args.scene_start, args.scene_start_samples)
    temporal_samples = load_samples(args.temporal, 1)
    target_shapes = {
        name: tuple(value.shape) for name, value in base_samples[0].items()
    }
    temporal, cropped = normalize_sample(temporal_samples[0], target_shapes)

    commands = [
        int(value) for value in args.temporal_commands.split(",") if value
    ]
    if commands and len(commands) != args.temporal_copies:
        raise ValueError("--temporal-commands must match --temporal-copies")
    if args.keep_scene_start > len(base_samples):
        raise ValueError("Not enough scene-start samples")

    selected = []
    lineage = []
    for index in range(args.keep_scene_start):
        selected.append(base_samples[index])
        lineage.append({"kind": "scene_start", "source_index": index})
    for index in range(args.temporal_copies):
        sample = {name: value.copy() for name, value in temporal.items()}
        if commands:
            if "command" not in sample:
                raise ValueError("Commands requested for an input set without command")
            sample["command"][...] = commands[index]
        selected.append(sample)
        lineage.append({
            "kind": "temporal",
            "source_index": 0,
            "command_override": commands[index] if commands else None,
        })

    arrays = {
        name: np.concatenate([sample[name] for sample in selected], axis=0)
        for name in target_shapes
    }
    nonfinite = tensor_audit(arrays)
    if nonfinite:
        raise ValueError("Non-finite calibration inputs: %s" % nonfinite)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **arrays)

    report = {
        "scene_start": {
            "path": os.path.abspath(args.scene_start),
            "sha256": sha256(args.scene_start),
            "available_samples": args.scene_start_samples,
            "selected_samples": args.keep_scene_start,
        },
        "temporal": {
            "path": os.path.abspath(args.temporal),
            "sha256": sha256(args.temporal),
            "selected_samples": args.temporal_copies,
            "cropped_inputs": cropped,
        },
        "lineage": lineage,
        "output": {
            "path": output_path,
            "bytes": os.path.getsize(output_path),
            "sha256": sha256(output_path),
            "samples": len(selected),
            "single_shapes": {
                name: list(shape) for name, shape in target_shapes.items()
            },
            "stored_shapes": {
                name: list(value.shape) for name, value in arrays.items()
            },
            "calibration_shapes": calibration_shapes(target_shapes),
            "nonfinite": nonfinite,
            "use_prev_bev": (
                np.asarray(arrays["use_prev_bev"]).reshape(-1).tolist()
                if "use_prev_bev" in arrays else None
            ),
            "commands": (
                np.asarray(arrays["command"]).reshape(-1).tolist()
                if "command" in arrays else None
            ),
            "prev_bev_range": {
                "min": float(arrays["prev_bev"].min()),
                "max": float(arrays["prev_bev"].max()),
                "nonzero": int(np.count_nonzero(arrays["prev_bev"])),
            },
        },
    }
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
