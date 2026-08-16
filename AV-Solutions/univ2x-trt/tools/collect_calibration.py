import argparse
import collections
import copy
import json
import os
import os.path as osp
import sys

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mmcv
import numpy as np
import torch
from torch.utils.data import Subset
from mmcv.fileio.file_client import HardDiskBackend
from mmdet3d.datasets import build_dataset

from cooperative_runtime import AgentState, agent_tensor, prepare_cooperative_inputs
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from runtime_common import load_config
from trt_engine import TensorRTEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("infrastructure_engine")
    parser.add_argument("ego_engine")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--infrastructure-target-prev", type=int)
    parser.add_argument("--ego-target-prev", type=int)
    parser.add_argument("--ego-target-coop", type=int)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--infrastructure-output")
    parser.add_argument("--ego-output")
    parser.add_argument("--without-labels", action="store_true")
    parser.add_argument("--command-file")
    parser.add_argument("--command-cycle", default="")
    parser.add_argument("--scene-start-only", action="store_true")
    parser.add_argument("--dataset-split", choices=("train", "test"), default="test")
    parser.add_argument("--sample-fraction", type=float, default=0.0)
    parser.add_argument("--sampling-strategy", choices=("sequential", "scene_uniform"),
                        default="sequential")
    parser.add_argument("--infrastructure-manifest-dir")
    parser.add_argument("--ego-manifest-dir")
    return parser.parse_args()


def patch_disk_backend_file_objects():
    original_get = HardDiskBackend.get

    def compatible_get(backend, filepath):
        if hasattr(filepath, "read"):
            return filepath.read()
        return original_get(backend, filepath)

    HardDiskBackend.get = compatible_get


def configure_label_free_pipeline(cfg, command_file):
    if command_file is None:
        raise ValueError("--without-labels requires --command-file")
    pipeline = copy.deepcopy(cfg.data.test.pipeline)
    label_transforms = {"LoadAnnotations3D_E2E", "GenerateOccFlowLabels"}
    pipeline = [
        transform for transform in pipeline
        if transform.get("type") not in label_transforms
    ]
    required_keys = {
        "veh2inf_rt", "img", "timestamp", "l2g_r_mat", "l2g_t", "command"
    }
    for transform in pipeline:
        if transform.get("type") != "MultiScaleFlipAug3D":
            continue
        for nested in transform.get("transforms", []):
            if nested.get("type") == "CustomCollect3D":
                nested["keys"] = [
                    key for key in nested.get("keys", []) if key in required_keys
                ]
    cfg.data.test.pipeline = pipeline
    cfg.data.test.inference_wo_label = True
    cfg.data.test.command_file = os.path.abspath(command_file)


def filtered_inputs(engine, values):
    return {name: values[name] for name in engine.input_names}


def append_sample(storage, inputs):
    for name, value in inputs.items():
        storage.setdefault(name, []).append(
            value.detach().cpu().numpy().copy()
        )


def save_samples(path, storage):
    if not storage:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {
        name: np.concatenate(values, axis=0)
        for name, values in storage.items()
    }


def select_scene_uniform_indices(dataset, target_count):
    scenes = collections.OrderedDict()
    for index, info in enumerate(dataset.data_infos):
        scenes.setdefault(str(info.get("scene_token")), []).append(index)
    if target_count < len(scenes):
        raise ValueError(
            "scene_uniform needs at least one frame per scene: %d < %d"
            % (target_count, len(scenes))
        )
    base, remainder = divmod(target_count, len(scenes))
    selected = []
    for scene_index, indices in enumerate(scenes.values()):
        count = min(len(indices), base + int(scene_index < remainder))
        if count == len(indices):
            selected.extend(indices)
        else:
            positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
            selected.extend(indices[int(position)] for position in positions)
    if len(selected) < target_count:
        remaining = sorted(set(range(len(dataset))) - set(selected))
        selected.extend(remaining[:target_count - len(selected)])
    return sorted(selected[:target_count])


def write_manifest_sample(directory, sample_number, dataset_index, inputs):
    os.makedirs(directory, exist_ok=True)
    filename = "sample_%04d_dataset_%06d.npz" % (sample_number, dataset_index)
    path = os.path.join(directory, filename)
    arrays = {
        name: value.detach().cpu().numpy().copy()
        for name, value in inputs.items()
    }
    np.savez(path, **arrays)
    return {
        "file": filename,
        "dataset_index": int(dataset_index),
        "bytes": os.path.getsize(path),
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "use_prev_bev": int(arrays["use_prev_bev"].reshape(-1)[0]),
        "command": (
            int(arrays["command"].reshape(-1)[0]) if "command" in arrays else None
        ),
        "track_count": int(arrays["prev_track_intances0"].shape[0]),
        "coop_track_count": (
            int(arrays["coop_track_intances0"].shape[0])
            if "coop_track_intances0" in arrays else None
        ),
    }


def save_manifest(directory, dataset_split, selected_indices, entries):
    if directory is None:
        return None
    manifest = {
        "schema_version": 1,
        "dataset_split": dataset_split,
        "selected_indices": [int(index) for index in selected_indices],
        "samples": entries,
        "sample_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    path = os.path.join(directory, "manifest.json")
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return {
        "directory": os.path.abspath(directory),
        "manifest": os.path.abspath(path),
        "samples": len(entries),
        "bytes": manifest["total_bytes"],
    }
    np.savez(path, **arrays)
    single_shapes = {
        name: list(values[0].shape)
        for name, values in storage.items()
    }
    calibration_shapes = ",".join(
        "%s:%s" % (name, "x".join(str(dim) for dim in shape))
        for name, shape in single_shapes.items()
    )
    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "samples": len(next(iter(storage.values()))),
        "single_shapes": single_shapes,
        "stored_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "calibration_shapes": calibration_shapes,
    }


def nonfinite_tensors(outputs):
    return {
        name: int((~torch.isfinite(value)).sum().item())
        for name, value in outputs.items()
        if value.is_floating_point() and not torch.isfinite(value).all()
    }


def tensor_range(value):
    if not value.is_floating_point() or value.numel() == 0:
        return None
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return {"min": None, "max": None}
    return {
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


def main():
    args = parse_args()
    command_cycle = [
        int(value) for value in args.command_cycle.split(",") if value
    ]
    patch_disk_backend_file_objects()
    cfg = load_config(args.config, workers=args.workers)
    if args.dataset_split == "train":
        train_ann_file = cfg.data.train.ann_file
        cfg.data.test = copy.deepcopy(cfg.data.test)
        cfg.data.test.ann_file = train_ann_file
        cfg.data.test.test_mode = True
    if args.without_labels:
        configure_label_free_pipeline(cfg, args.command_file)
    dataset = build_dataset(cfg.data.test)
    total_dataset_frames = len(dataset)
    selected_indices = list(range(total_dataset_frames))
    if args.scene_start_only:
        selected_indices = [
            index for index, info in enumerate(dataset.data_infos)
            if int(info.get("frame_idx", -1)) == 0
        ]
    elif args.sample_fraction:
        if not 0.0 < args.sample_fraction <= 1.0:
            raise ValueError("--sample-fraction must be in (0, 1]")
        target_count = max(1, int(round(total_dataset_frames * args.sample_fraction)))
        if args.sampling_strategy == "scene_uniform":
            selected_indices = select_scene_uniform_indices(dataset, target_count)
        else:
            selected_indices = list(range(target_count))
    dataset_for_loader = (
        dataset if len(selected_indices) == total_dataset_frames
        else Subset(dataset, selected_indices)
    )
    loader = build_dataloader(
        dataset_for_loader,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    infrastructure_engine = TensorRTEngine(args.infrastructure_engine, args.plugin)
    ego_engine = TensorRTEngine(args.ego_engine, args.plugin)
    device = torch.device("cuda")
    infrastructure_state = AgentState(device)
    ego_state = AgentState(device)
    execution_stream = torch.cuda.Stream()

    candidate_frames = len(dataset_for_loader)
    frame_limit = candidate_frames if args.max_frames <= 0 else min(
        args.max_frames, candidate_frames
    )
    shape_counts = collections.Counter()
    frames = []
    infrastructure_samples = {}
    ego_samples = {}
    infrastructure_manifest_entries = []
    ego_manifest_entries = []
    progress = mmcv.ProgressBar(frame_limit)

    for frame_index, data in enumerate(loader):
        if frame_index >= frame_limit:
            break
        ego_data = data["ego_agent_data"]
        infrastructure_data = data["other_agent_data_dict"]["model_other_agent_inf"]
        with torch.cuda.stream(execution_stream):
            infrastructure_values, _ = infrastructure_state.build_inputs(
                infrastructure_data,
                include_ground_truth=not args.without_labels,
            )
            infrastructure_inputs = filtered_inputs(
                infrastructure_engine, infrastructure_values
            )
            infrastructure_input_nonfinite = nonfinite_tensors(
                infrastructure_inputs
            )
            infrastructure_prev = int(
                infrastructure_inputs["prev_track_intances0"].shape[0]
            )
            collect_infrastructure = (
                (args.infrastructure_output is not None
                 or args.infrastructure_manifest_dir is not None)
                and (args.infrastructure_target_prev is None
                     or args.infrastructure_target_prev == infrastructure_prev)
                and max(
                    len(next(iter(infrastructure_samples.values()), [])),
                    len(infrastructure_manifest_entries),
                ) < args.max_samples
            )
            if collect_infrastructure:
                if args.infrastructure_manifest_dir:
                    infrastructure_manifest_entries.append(write_manifest_sample(
                        args.infrastructure_manifest_dir,
                        len(infrastructure_manifest_entries),
                        selected_indices[frame_index],
                        infrastructure_inputs,
                    ))
                else:
                    append_sample(infrastructure_samples, infrastructure_inputs)
            infrastructure_outputs = infrastructure_engine.infer(
                infrastructure_inputs, synchronize=False
            )
            execution_stream.synchronize()
            infrastructure_nonfinite = nonfinite_tensors(infrastructure_outputs)
            infrastructure_state.update(
                infrastructure_outputs,
                timestamp_fallback=infrastructure_inputs["timestamp"],
            )

            ego_values, _ = ego_state.build_inputs(
                ego_data,
                include_ground_truth=not args.without_labels,
            )
            if command_cycle:
                ego_values["command"] = torch.tensor(
                    [command_cycle[frame_index % len(command_cycle)]],
                    dtype=torch.int64,
                    device=device,
                )
            # Match MultiAgent.forward_test: obtain the cooperative transform
            # from the infrastructure-side sample, not the ego identity value.
            veh2inf_rt = agent_tensor(
                infrastructure_data, "veh2inf_rt", device, torch.float32
            )
            cooperative_inputs, cooperative_stats = prepare_cooperative_inputs(
                infrastructure_outputs, ego_state, veh2inf_rt
            )
            ego_values.update(cooperative_inputs)
            ego_inputs = filtered_inputs(ego_engine, ego_values)
            ego_input_nonfinite = nonfinite_tensors(ego_inputs)
            ego_prev = int(ego_inputs["prev_track_intances0"].shape[0])
            coop_count = int(ego_inputs["coop_track_intances0"].shape[0])
            collect_ego = (
                (args.ego_output is not None or args.ego_manifest_dir is not None)
                and (args.ego_target_prev is None or args.ego_target_prev == ego_prev)
                and (args.ego_target_coop is None or args.ego_target_coop == coop_count)
                and max(
                    len(next(iter(ego_samples.values()), [])),
                    len(ego_manifest_entries),
                ) < args.max_samples
            )
            if collect_ego:
                if args.ego_manifest_dir:
                    ego_manifest_entries.append(write_manifest_sample(
                        args.ego_manifest_dir,
                        len(ego_manifest_entries),
                        selected_indices[frame_index],
                        ego_inputs,
                    ))
                else:
                    append_sample(ego_samples, ego_inputs)
            ego_outputs = ego_engine.infer(ego_inputs, synchronize=False)
            execution_stream.synchronize()
            ego_nonfinite = nonfinite_tensors(ego_outputs)
            ego_state.update(
                ego_outputs,
                timestamp_fallback=ego_inputs["timestamp"],
            )

        key = (infrastructure_prev, ego_prev, coop_count)
        shape_counts[key] += 1
        frames.append({
            "frame": frame_index,
            "dataset_index": selected_indices[frame_index],
            "infrastructure_prev": infrastructure_prev,
            "ego_prev": ego_prev,
            "coop_count": coop_count,
            "infrastructure_next": int(
                infrastructure_outputs["prev_track_intances0_out"].shape[0]
            ),
            "ego_next": int(ego_outputs["prev_track_intances0_out"].shape[0]),
            "infrastructure_input_nonfinite": infrastructure_input_nonfinite,
            "ego_input_nonfinite": ego_input_nonfinite,
            "infrastructure_use_prev_bev": int(
                infrastructure_inputs["use_prev_bev"].reshape(-1)[0].item()
            ),
            "ego_use_prev_bev": int(
                ego_inputs["use_prev_bev"].reshape(-1)[0].item()
            ),
            "infrastructure_prev_bev_range": tensor_range(
                infrastructure_inputs["prev_bev"]
            ),
            "ego_prev_bev_range": tensor_range(ego_inputs["prev_bev"]),
            "infrastructure_nonfinite": infrastructure_nonfinite,
            "ego_nonfinite": ego_nonfinite,
            **cooperative_stats,
        })
        progress.update()

        infrastructure_done = (
            (args.infrastructure_output is None
             and args.infrastructure_manifest_dir is None)
            or max(
                len(next(iter(infrastructure_samples.values()), [])),
                len(infrastructure_manifest_entries),
            ) >= args.max_samples
        )
        ego_done = (
            (args.ego_output is None and args.ego_manifest_dir is None)
            or max(
                len(next(iter(ego_samples.values()), [])),
                len(ego_manifest_entries),
            ) >= args.max_samples
        )
        if (
            args.infrastructure_output or args.ego_output
            or args.infrastructure_manifest_dir or args.ego_manifest_dir
        ) and infrastructure_done and ego_done:
            break

    infrastructure_artifact = save_samples(
        args.infrastructure_output, infrastructure_samples
    ) if args.infrastructure_output else None
    ego_artifact = save_samples(args.ego_output, ego_samples) if args.ego_output else None
    infrastructure_manifest = save_manifest(
        args.infrastructure_manifest_dir,
        args.dataset_split,
        selected_indices,
        infrastructure_manifest_entries,
    )
    ego_manifest = save_manifest(
        args.ego_manifest_dir,
        args.dataset_split,
        selected_indices,
        ego_manifest_entries,
    )
    report = {
        "dataset_split": args.dataset_split,
        "dataset_frames": total_dataset_frames,
        "candidate_frames": candidate_frames,
        "scene_start_only": args.scene_start_only,
        "sample_fraction": args.sample_fraction,
        "sampling_strategy": args.sampling_strategy,
        "selected_indices": selected_indices,
        "processed_frames": len(frames),
        "shape_counts": [
            {
                "infrastructure_prev": key[0],
                "ego_prev": key[1],
                "coop_count": key[2],
                "count": count,
            }
            for key, count in shape_counts.most_common()
        ],
        "frames": frames,
        "infrastructure_calibration": infrastructure_artifact,
        "ego_calibration": ego_artifact,
        "infrastructure_manifest": infrastructure_manifest,
        "ego_manifest": ego_manifest,
        "without_labels": args.without_labels,
        "command_file": os.path.abspath(args.command_file)
        if args.command_file else None,
        "command_cycle": command_cycle,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "processed_frames": report["processed_frames"],
        "most_common_shapes": report["shape_counts"][:10],
        "infrastructure_calibration": infrastructure_artifact,
        "ego_calibration": ego_artifact,
        "infrastructure_manifest": infrastructure_manifest,
        "ego_manifest": ego_manifest,
        "report": os.path.abspath(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()
