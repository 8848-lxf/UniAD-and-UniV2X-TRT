#!/usr/bin/env python3

"""Audit UniAD deployment BFGS against the public CasADi/IPOPT post-process."""

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


PRECISIONS = {
    "fp32": "fp32",
    "fp16": "fp16",
    "int8": "int8_official168_occ_terminal_fp16",
}
METHODS = ("bfgs_float", "ipopt_float", "ipopt_original")
N_FUTURE = 6
OCC_SHAPE = (5, 50, 50)
OCC_VALUES = int(np.prod(OCC_SHAPE))
PACKED_BYTES = (OCC_VALUES + 7) // 8

_WORKER_RAWS = None
_WORKER_PACKED = None
_WORKER_OPTIMIZER = None
_WORKER_BFGS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--official-optimizer-source", required=True, type=Path)
    parser.add_argument("--deployment-postprocess-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load %s from %s" % (name, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trajectory_csv(path):
    expected = [
        axis + str(step)
        for step in range(1, N_FUTURE + 1)
        for axis in ("x", "y")
    ]
    rows = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["frame"] + expected:
            raise ValueError("Unexpected columns in %s: %r" % (path, reader.fieldnames))
        for row in reader:
            frame = int(row["frame"])
            values = np.asarray([float(row[key]) for key in expected], dtype=np.float32)
            rows[frame] = values.reshape(N_FUTURE, 2)
    if sorted(rows) != list(range(len(rows))):
        raise ValueError("Frames in %s are not contiguous from zero" % path)
    return np.stack([rows[index] for index in range(len(rows))])


def read_manifest(path):
    with path.open() as handle:
        return json.load(handle)


def validate_inputs(args):
    details = {}
    common_frames = None
    common_protocol = None
    for precision, directory in PRECISIONS.items():
        root = args.artifact_root / directory
        raw_path = root / "planning_predictions_raw.csv"
        raw_manifest_path = Path(str(raw_path) + ".manifest.json")
        occupancy_path = root / "seg_out.packbits"
        occupancy_manifest_path = Path(str(occupancy_path) + ".manifest.json")
        for path in (raw_path, raw_manifest_path, occupancy_path, occupancy_manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        raw_manifest = read_manifest(raw_manifest_path)
        occupancy_manifest = read_manifest(occupancy_manifest_path)
        frames = int(raw_manifest["frames"])
        protocol = raw_manifest.get("temporal_protocol")
        if int(occupancy_manifest["frames"]) != frames:
            raise ValueError("Frame mismatch for %s" % precision)
        if occupancy_manifest.get("temporal_protocol") != protocol:
            raise ValueError("Protocol mismatch for %s" % precision)
        if tuple(occupancy_manifest["shape"]) != (1, 5, 1, 50, 50):
            raise ValueError("Unexpected occupancy shape for %s" % precision)
        if occupancy_path.stat().st_size != frames * PACKED_BYTES:
            raise ValueError("Unexpected packed occupancy size for %s" % precision)
        if common_frames is None:
            common_frames = frames
            common_protocol = protocol
        elif frames != common_frames or protocol != common_protocol:
            raise ValueError("Precision inputs do not share one frame/protocol contract")
        details[precision] = {
            "directory": directory,
            "raw_path": str(raw_path),
            "raw_manifest": raw_manifest,
            "occupancy_path": str(occupancy_path),
            "occupancy_manifest": occupancy_manifest,
        }
    frames = common_frames
    if args.max_frames > 0:
        frames = min(frames, args.max_frames)
    for category in ("sdc_planning", "sdc_planning_mask", "gt_segmentation"):
        path = args.ground_truth / category
        if not path.is_dir() or not (path / "0.npy").is_file():
            raise FileNotFoundError(path)
        if not (path / (str(frames - 1) + ".npy")).is_file():
            raise FileNotFoundError(path / (str(frames - 1) + ".npy"))
    return frames, common_frames, common_protocol, details


def worker_init(raw_paths, occupancy_paths, optimizer_path, bfgs_path):
    global _WORKER_RAWS, _WORKER_PACKED, _WORKER_OPTIMIZER, _WORKER_BFGS
    _WORKER_RAWS = {
        precision: load_trajectory_csv(Path(path))
        for precision, path in raw_paths.items()
    }
    _WORKER_PACKED = {
        precision: np.memmap(
            path,
            mode="r",
            dtype=np.uint8,
            shape=(_WORKER_RAWS[precision].shape[0], PACKED_BYTES),
        )
        for precision, path in occupancy_paths.items()
    }
    _WORKER_OPTIMIZER = load_module(
        "official_collision_optimization_worker", Path(optimizer_path)
    ).CollisionNonlinearOptimizer
    _WORKER_BFGS = load_module(
        "deployment_planning_postprocess_worker", Path(bfgs_path)
    ).optimize_collision


def unpack_occupancy(precision, frame):
    packed = np.asarray(_WORKER_PACKED[precision][frame])
    bits = np.unpackbits(packed, bitorder="little")[:OCC_VALUES]
    return bits.reshape(OCC_SHAPE).astype(np.bool_, copy=False)


def occupied_by_timestep(reference, occupancy, original_integer_coordinates):
    horizon, height, width = occupancy.shape
    result = []
    total = 0
    for timestep, point in enumerate(reference):
        occupancy_timestep = min(timestep + 1, horizon - 1)
        pixels = np.argwhere(occupancy[occupancy_timestep])
        if not pixels.size:
            occupied = np.empty((0, 2), dtype=np.float64)
        else:
            x = (pixels[:, 1] - height // 2) * 0.5 + 0.25
            y = (pixels[:, 0] - width // 2) * 0.5 + 0.25
            if original_integer_coordinates:
                # planning_head.py writes these values back into torch.nonzero's
                # int64 tensor, which truncates toward zero before filtering.
                occupied = np.stack([x, y], axis=-1).astype(np.int64)
            else:
                occupied = np.stack([x, y], axis=-1).astype(np.float64)
            delta = point[None, :2] - occupied[:, :2]
            occupied = occupied[np.sum(delta * delta, axis=1) < 25.0]
        result.append(occupied)
        total += int(occupied.shape[0])
    return result, total


def optimize_ipopt(reference, occupancy, original_integer_coordinates):
    occupied, candidates = occupied_by_timestep(
        reference, occupancy, original_integer_coordinates
    )
    if candidates == 0:
        return reference.astype(np.float32, copy=True), candidates
    optimizer = _WORKER_OPTIMIZER(
        N_FUTURE, 0.5, 1.0, 5.0, occupied
    )
    optimizer.set_reference_trajectory(reference)
    solution = optimizer.solve()
    optimized = np.stack(
        [
            solution.value(optimizer.position_x),
            solution.value(optimizer.position_y),
        ],
        axis=-1,
    )
    return optimized.astype(np.float32), candidates


def run_task(task):
    raw_precision, occupancy_precision, frame, methods = task
    reference = _WORKER_RAWS[raw_precision][frame].astype(np.float64)
    occupancy = unpack_occupancy(occupancy_precision, frame)
    outputs = {}
    for method in methods:
        try:
            if method == "bfgs_float":
                optimized, statistics = _WORKER_BFGS(reference, occupancy)
                candidates = int(statistics["collision_occupied_points"])
            elif method == "ipopt_float":
                optimized, candidates = optimize_ipopt(reference, occupancy, False)
            elif method == "ipopt_original":
                optimized, candidates = optimize_ipopt(reference, occupancy, True)
            else:
                raise ValueError("Unknown method: %s" % method)
            if optimized.shape != (N_FUTURE, 2) or not np.isfinite(optimized).all():
                raise ValueError("Non-finite or malformed optimized trajectory")
            outputs[method] = {
                "status": 1,
                "trajectory": optimized,
                "candidates": candidates,
                "error": "",
            }
        except Exception as error:  # retain the failing frame for audit
            outputs[method] = {
                "status": 2,
                "trajectory": reference.astype(np.float32),
                "candidates": -1,
                "error": "%s: %s" % (type(error).__name__, error),
            }
    return raw_precision, occupancy_precision, frame, outputs


def open_result_arrays(output_dir, method, raw_precision, occupancy_precision, frames):
    root = output_dir / method / (raw_precision + "_raw__" + occupancy_precision + "_occ")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trajectory": root / "trajectories.npy",
        "status": root / "status.npy",
        "candidates": root / "candidate_points.npy",
    }
    arrays = {}
    specifications = {
        "trajectory": (np.float32, (frames, N_FUTURE, 2)),
        "status": (np.uint8, (frames,)),
        "candidates": (np.int32, (frames,)),
    }
    for name, (dtype, shape) in specifications.items():
        if paths[name].is_file():
            value = np.lib.format.open_memmap(paths[name], mode="r+")
            if value.dtype != dtype or value.shape != shape:
                raise ValueError("Resume array contract mismatch: %s" % paths[name])
        else:
            value = np.lib.format.open_memmap(
                paths[name], mode="w+", dtype=dtype, shape=shape
            )
            value[...] = 0
            value.flush()
        arrays[name] = value
    return root, arrays


def pdt_timestamp():
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    if hasattr(time, "tzset"):
        time.tzset()
    value = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if previous is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous
    if hasattr(time, "tzset"):
        time.tzset()
    return value


def load_ground_truth(root, frames):
    targets = []
    masks = []
    segmentations = []
    for frame in range(frames):
        targets.append(np.load(root / "sdc_planning" / (str(frame) + ".npy"))[0, 0, :, :2])
        masks.append(np.load(root / "sdc_planning_mask" / (str(frame) + ".npy"))[0, 0, :, :2])
        segmentations.append(np.load(root / "gt_segmentation" / (str(frame) + ".npy"))[0, 1:7])
    return (
        np.asarray(targets, dtype=np.float32),
        np.asarray(masks, dtype=np.float32),
        np.asarray(segmentations, dtype=np.bool_),
    )


def footprint_offsets():
    from skimage.draw import polygon

    dx = np.asarray([0.5, 0.5], dtype=np.float32)
    bx = np.asarray([-12.25, -12.25], dtype=np.float32)
    points = np.asarray(
        [
            [-4.084 / 2.0 + 0.5, 1.85 / 2.0],
            [4.084 / 2.0 + 0.5, 1.85 / 2.0],
            [4.084 / 2.0 + 0.5, -1.85 / 2.0],
            [-4.084 / 2.0 + 0.5, -1.85 / 2.0],
        ]
    )
    points = (points - bx) / dx
    points[:, [0, 1]] = points[:, [1, 0]]
    rows, columns = polygon(points[:, 1], points[:, 0])
    return dx, bx, np.stack([rows, columns], axis=-1)


def footprint_collision(trajectory, segmentation, dx, footprint):
    swapped = trajectory[:, [1, 0]] / dx
    pixels = swapped[:, None, :] + footprint[None, :, :]
    # Preserve UniAD PlanningMetric's published boundary behavior exactly.
    rows = np.clip(pixels[:, :, 0].astype(np.int32), 0, 49)
    columns = np.clip(pixels[:, :, 1].astype(np.int32), 0, 49)
    result = np.zeros(N_FUTURE, dtype=np.bool_)
    for timestep in range(N_FUTURE):
        valid = (
            (rows[timestep] >= 0)
            & (rows[timestep] < 50)
            & (columns[timestep] >= 0)
            & (columns[timestep] < 50)
        )
        result[timestep] = np.any(
            segmentation[timestep, rows[timestep, valid], columns[timestep, valid]]
        )
    return result


def planning_metrics(predictions, targets, masks, segmentations):
    dx, bx, footprint = footprint_offsets()
    l2_sum = np.zeros(N_FUTURE, dtype=np.float64)
    point_sum = np.zeros(N_FUTURE, dtype=np.float64)
    box_sum = np.zeros(N_FUTURE, dtype=np.float64)
    for prediction, target, mask, segmentation in zip(
        predictions, targets, masks, segmentations
    ):
        l2_sum += np.sqrt(np.sum((prediction - target) ** 2 * mask, axis=-1))
        target_collision = footprint_collision(target, segmentation, dx, footprint)
        row = ((prediction[:, 1] - bx[0]) / dx[0]).astype(np.int64)
        column = ((prediction[:, 0] - bx[1]) / dx[1]).astype(np.int64)
        valid = (
            (row >= 0) & (row < 50) & (column >= 0) & (column < 50)
            & ~target_collision
        )
        timesteps = np.arange(N_FUTURE)[valid]
        point = np.zeros(N_FUTURE, dtype=np.float64)
        point[valid] = segmentation[timesteps, row[valid], column[valid]]
        box = footprint_collision(prediction, segmentation, dx, footprint).astype(np.float64)
        box[target_collision] = 0.0
        point_sum += point
        box_sum += box
    count = predictions.shape[0]
    l2 = l2_sum / count
    point = point_sum / count * 100.0
    box = box_sum / count * 100.0
    return {
        "avg_l2_m": float(l2.mean()),
        "l2_by_timestep_m": l2.tolist(),
        "avg_point_collision_percent": float(point.mean()),
        "point_collision_by_timestep_percent": point.tolist(),
        "avg_box_collision_percent": float(box.mean()),
        "box_collision_by_timestep_percent": box.tolist(),
    }


def delta_summary(candidate, reference):
    point_delta = np.linalg.norm(candidate - reference, axis=-1)
    frame_delta = point_delta.max(axis=1)
    return {
        "mean_point_l2_m": float(point_delta.mean()),
        "p50_point_l2_m": float(np.percentile(point_delta, 50)),
        "p95_point_l2_m": float(np.percentile(point_delta, 95)),
        "p99_point_l2_m": float(np.percentile(point_delta, 99)),
        "max_point_l2_m": float(point_delta.max()),
        "frames_over_1e_7_m": int(np.count_nonzero(frame_delta > 1.0e-7)),
        "frames_over_1cm": int(np.count_nonzero(frame_delta > 0.01)),
        "frames_over_10cm": int(np.count_nonzero(frame_delta > 0.1)),
        "max_frame": int(np.argmax(frame_delta)),
    }


def write_csv(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(args, frames, total_frames, protocol, input_details, stores, raws):
    targets, masks, segmentations = load_ground_truth(args.ground_truth, frames)
    metrics = {}
    for method in METHODS:
        metrics[method] = {}
        for raw_precision in PRECISIONS:
            for occupancy_precision in PRECISIONS:
                key = raw_precision + "_raw__" + occupancy_precision + "_occ"
                arrays = stores[(method, raw_precision, occupancy_precision)][1]
                statuses = np.asarray(arrays["status"])
                if np.any(statuses != 1):
                    raise RuntimeError(
                        "%s/%s has %d incomplete or failed frames"
                        % (method, key, np.count_nonzero(statuses != 1))
                    )
                predictions = np.asarray(arrays["trajectory"])
                raw = raws[raw_precision][:frames]
                displacement = np.linalg.norm(predictions - raw, axis=-1)
                entry = planning_metrics(
                    predictions, targets, masks, segmentations
                )
                entry.update({
                    "candidate_points": int(np.asarray(arrays["candidates"]).sum()),
                    "frames_modified": int(np.count_nonzero(np.max(displacement, axis=1) > 1.0e-7)),
                    "points_modified": int(np.count_nonzero(displacement > 1.0e-7)),
                    "max_raw_to_optimized_delta_m": float(displacement.max()),
                })
                metrics[method][key] = entry

    comparisons = {}
    for raw_precision in PRECISIONS:
        for occupancy_precision in PRECISIONS:
            key = raw_precision + "_raw__" + occupancy_precision + "_occ"
            bfgs = np.asarray(stores[("bfgs_float", raw_precision, occupancy_precision)][1]["trajectory"])
            ipopt_float = np.asarray(stores[("ipopt_float", raw_precision, occupancy_precision)][1]["trajectory"])
            ipopt_original = np.asarray(stores[("ipopt_original", raw_precision, occupancy_precision)][1]["trajectory"])
            comparisons[key] = {
                "bfgs_float_vs_ipopt_float_solver_only": delta_summary(bfgs, ipopt_float),
                "ipopt_float_vs_ipopt_original_coordinate_only": delta_summary(ipopt_float, ipopt_original),
                "bfgs_float_vs_ipopt_original_total": delta_summary(bfgs, ipopt_original),
            }

    cpp_checks = {}
    for precision, directory in PRECISIONS.items():
        cpp = load_trajectory_csv(args.artifact_root / directory / "planning_predictions.csv")[:frames]
        bfgs = np.asarray(stores[("bfgs_float", precision, precision)][1]["trajectory"])
        cpp_checks[precision] = delta_summary(bfgs, cpp)

    summary = {
        "schema_version": 1,
        "timestamp_pdt": pdt_timestamp(),
        "frames": frames,
        "source_validation_frames": total_frames,
        "temporal_protocol": protocol,
        "artifact_root": str(args.artifact_root),
        "ground_truth": str(args.ground_truth),
        "official_optimizer_source": str(args.official_optimizer_source),
        "deployment_postprocess_source": str(args.deployment_postprocess_source),
        "methods": {
            "bfgs_float": "current C++-matching independent-point BFGS with floating grid centers",
            "ipopt_float": "public CollisionNonlinearOptimizer with floating grid centers; isolates solver",
            "ipopt_original": "public CollisionNonlinearOptimizer with planning_head.py int64 write-back semantics",
        },
        "input_contract": input_details,
        "cpp_bfgs_reproduction": cpp_checks,
        "metrics": metrics,
        "trajectory_comparisons": comparisons,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    cross_rows = []
    for method in METHODS:
        for raw_precision in PRECISIONS:
            for occupancy_precision in PRECISIONS:
                key = raw_precision + "_raw__" + occupancy_precision + "_occ"
                entry = metrics[method][key]
                cross_rows.append({
                    "method": method,
                    "raw_precision": raw_precision,
                    "occupancy_precision": occupancy_precision,
                    "avg_l2_m": entry["avg_l2_m"],
                    "avg_point_collision_percent": entry["avg_point_collision_percent"],
                    "avg_box_collision_percent": entry["avg_box_collision_percent"],
                    "candidate_points": entry["candidate_points"],
                    "frames_modified": entry["frames_modified"],
                    "points_modified": entry["points_modified"],
                    "max_raw_to_optimized_delta_m": entry["max_raw_to_optimized_delta_m"],
                })
    write_csv(
        args.output_dir / "cross_ablation.csv",
        cross_rows,
        list(cross_rows[0]),
    )
    return summary_path


def main():
    args = parse_args()
    frames, total_frames, protocol, input_details = validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {
        precision: details["raw_path"]
        for precision, details in input_details.items()
    }
    occupancy_paths = {
        precision: details["occupancy_path"]
        for precision, details in input_details.items()
    }
    raws = {
        precision: load_trajectory_csv(Path(path))
        for precision, path in raw_paths.items()
    }

    stores = {}
    tasks = []
    for method in METHODS:
        for raw_precision in PRECISIONS:
            for occupancy_precision in PRECISIONS:
                store = open_result_arrays(
                    args.output_dir, method, raw_precision, occupancy_precision, frames
                )
                stores[(method, raw_precision, occupancy_precision)] = store
    for raw_precision in PRECISIONS:
        for occupancy_precision in PRECISIONS:
            for frame in range(frames):
                pending = [
                    method
                    for method in METHODS
                    if stores[(method, raw_precision, occupancy_precision)][1]["status"][frame] != 1
                ]
                if pending:
                    tasks.append((raw_precision, occupancy_precision, frame, pending))

    started = time.time()
    errors = []
    if tasks:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            mp_context=context,
            initializer=worker_init,
            initargs=(
                raw_paths,
                occupancy_paths,
                str(args.official_optimizer_source),
                str(args.deployment_postprocess_source),
            ),
        ) as pool:
            futures = [pool.submit(run_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), 1):
                raw_precision, occupancy_precision, frame, outputs = future.result()
                for method, output in outputs.items():
                    arrays = stores[(method, raw_precision, occupancy_precision)][1]
                    arrays["trajectory"][frame] = output["trajectory"]
                    arrays["candidates"][frame] = output["candidates"]
                    arrays["status"][frame] = output["status"]
                    if output["status"] != 1:
                        errors.append({
                            "method": method,
                            "raw_precision": raw_precision,
                            "occupancy_precision": occupancy_precision,
                            "frame": frame,
                            "error": output["error"],
                        })
                if completed % args.progress_every == 0 or completed == len(tasks):
                    elapsed = time.time() - started
                    print(
                        "completed %d/%d tasks in %.1fs (%.2f tasks/s)"
                        % (completed, len(tasks), elapsed, completed / max(elapsed, 1.0)),
                        flush=True,
                    )
                    for _, arrays in stores.values():
                        for value in arrays.values():
                            value.flush()
    for _, arrays in stores.values():
        for value in arrays.values():
            value.flush()
    if errors:
        with (args.output_dir / "errors.json").open("w") as handle:
            json.dump(errors, handle, indent=2, sort_keys=True)
            handle.write("\n")
        raise RuntimeError("%d optimizer tasks failed; see errors.json" % len(errors))
    summary_path = summarize(
        args, frames, total_frames, protocol, input_details, stores, raws
    )
    print("wrote %s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
