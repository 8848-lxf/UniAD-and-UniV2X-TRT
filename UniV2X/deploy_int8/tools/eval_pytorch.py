import argparse
import csv
import json
import os
import os.path as osp
import time

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.dense_heads.occ_head_plugin import (
    IntersectionOverUnion,
    PanopticMetric,
)
from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head_plugin import (
    PlanningMetric,
)

from runtime_common import build_dataset_and_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def scalarize(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: scalarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalarize(item) for item in value]
    return value


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p99_ms": float(np.percentile(array, 99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "fps_from_mean": float(1000.0 / array.mean()),
    }


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.output_dir)
    cfg, dataset, model, checkpoint = build_dataset_and_model(
        args.config, args.checkpoint, workers=args.workers
    )
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])

    eval_occ = bool(getattr(model.module.model_ego_agent, "with_occ_head", False))
    eval_planning = bool(
        getattr(model.module.model_ego_agent, "with_planning_head", False)
    )
    ranges = {"30x30": (70, 130), "100x100": (0, 200)}
    iou_metrics = {
        key: IntersectionOverUnion(2).cuda() for key in ranges
    } if eval_occ else {}
    panoptic_metrics = {
        key: PanopticMetric(
            n_classes=2, temporally_consistent=True
        ).cuda()
        for key in ranges
    } if eval_occ else {}
    planning_metric = (
        PlanningMetric(n_future=dataset.planning_steps).cuda()
        if eval_planning
        else None
    )

    frame_limit = len(dataset) if args.max_frames <= 0 else min(
        args.max_frames, len(dataset)
    )
    outputs = []
    rows = []
    num_occ = 0
    iterator = iter(loader)
    progress = mmcv.ProgressBar(frame_limit)

    for frame_index in range(frame_limit):
        e2e_start = time.perf_counter()
        data_start = e2e_start
        data = next(iterator)
        data_end = time.perf_counter()

        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()

        if eval_planning:
            planning = result[0]["planning"]
            planning_gt = planning["planning_gt"]
            predicted = planning["result_planning"]["sdc_traj"]
            steps = dataset.planning_steps
            planning_metric(
                predicted[:, :steps, :2],
                planning_gt["sdc_planning"][0][0, :, :steps, :2],
                planning_gt["sdc_planning_mask"][0][0, :, :steps, :2],
                planning_gt["segmentation"][0][:, 1 : steps + 1],
                planning_gt["drivable_gt"],
            )
            result[0]["planning_traj"] = predicted
            result[0]["planning_traj_gt"] = planning_gt["sdc_planning"]
            result[0]["command"] = planning_gt["command"]

        if eval_occ:
            invalid = data["ego_agent_data"]["gt_occ_has_invalid_frame"][0]
            if not invalid.item() and "occ" in result[0]:
                num_occ += 1
                for key, (start, end) in ranges.items():
                    limit = slice(start, end)
                    occ = result[0]["occ"]
                    iou_metrics[key](
                        occ["seg_out"][..., limit, limit].contiguous(),
                        occ["seg_gt"][..., limit, limit].contiguous(),
                    )
                    panoptic_metrics[key](
                        occ["ins_seg_out"][..., limit, limit].contiguous().detach(),
                        occ["ins_seg_gt"][..., limit, limit].contiguous(),
                    )

        result[0].pop("occ", None)
        result[0].pop("planning", None)
        outputs.extend(result)
        e2e_end = time.perf_counter()
        rows.append(
            {
                "frame": frame_index,
                "data_ms": (data_end - data_start) * 1000.0,
                "forward_ms": (forward_end - forward_start) * 1000.0,
                "end_to_end_ms": (e2e_end - e2e_start) * 1000.0,
            }
        )
        progress.update()

    result_bundle = {"bbox_results": outputs}
    metrics = {}
    if eval_occ:
        occ_result = {}
        for key in ranges:
            panoptic = panoptic_metrics[key].compute()
            for metric_name, value in panoptic.items():
                occ_result.setdefault(metric_name, []).append(100 * value[1].item())
            occ_result.setdefault("iou", []).append(
                100 * iou_metrics[key].compute()[1].item()
            )
        occ_result["num_occ"] = num_occ
        occ_result["ratio_occ"] = num_occ / frame_limit
        result_bundle["occ_results_computed"] = occ_result
        metrics["occupancy"] = scalarize(occ_result)
    if eval_planning:
        planning_result = planning_metric.compute()
        result_bundle["planning_results_computed"] = planning_result
        metrics["planning"] = scalarize(planning_result)

    result_path = osp.join(args.output_dir, "results.pkl")
    mmcv.dump(result_bundle, result_path)
    with open(osp.join(args.output_dir, "task_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    with open(osp.join(args.output_dir, "latency_frames.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    warmup = min(args.warmup_frames, max(0, len(rows) - 1))
    measured = rows[warmup:]
    latency = {
        "schema_version": 1,
        "framework": "PyTorch",
        "precision": "FP32",
        "gpu": torch.cuda.get_device_name(0),
        "dataset_frames": frame_limit,
        "warmup_frames_excluded": warmup,
        "checkpoint_epoch": checkpoint.get("meta", {}).get("epoch"),
        "definitions": {
            "forward": "CUDA-synchronized wall time around cooperative MultiAgent forward",
            "end_to_end": "Dataloader wait, forward, occupancy/planning metric update, and result postprocessing",
        },
        "forward": summarize([row["forward_ms"] for row in measured]),
        "end_to_end": summarize([row["end_to_end_ms"] for row in measured]),
    }
    with open(osp.join(args.output_dir, "latency_metrics.json"), "w") as handle:
        json.dump(latency, handle, indent=2, allow_nan=False)

    if args.evaluate and frame_limit == len(dataset):
        eval_dir = osp.join(args.output_dir, "full_task_eval")
        mmcv.mkdir_or_exist(eval_dir)
        detail = dataset.evaluate(
            result_bundle,
            metric=["bbox"],
            jsonfile_prefix=eval_dir,
            pipeline=cfg.get("evaluation", {}).get("pipeline"),
        )
        with open(osp.join(args.output_dir, "dataset_metrics.json"), "w") as handle:
            json.dump(scalarize(detail), handle, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()
