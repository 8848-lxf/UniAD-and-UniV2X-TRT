import argparse
import json
import os
import os.path as osp
import sys
import time
import types

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
DEPLOY_ROOT = osp.join(REPO_ROOT, "deploy_int8", "UniV2X_deploy")
TRT_FUNCTIONS = osp.join(
    DEPLOY_ROOT, "projects", "mmdet3d_plugin", "uniad", "functions"
)
for source_root in (REPO_ROOT, DEPLOY_ROOT, TRT_FUNCTIONS):
    while source_root in sys.path:
        sys.path.remove(source_root)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TRT_FUNCTIONS)
sys.path.insert(0, DEPLOY_ROOT)

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.fileio.file_client import HardDiskBackend
from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS

import projects.mmdet3d_plugin  # noqa: E402,F401

from cooperative_runtime import AgentState, agent_tensor, prepare_cooperative_inputs
from projects.mmdet3d_plugin.datasets.builder import (
    build_dataloader,
)
from projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin import (
    IntersectionOverUnion,
    PanopticMetric,
)
from projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.utils import (
    predict_instance_segmentation_and_trajectories,
)
from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin import (
    PlanningMetric as BasePlanningMetric,
)
from projects.mmdet3d_plugin.uniad.dense_heads.planning_head import (
    PlanningHeadSingleMode,
)
from trt_runtime import (
    EGO_INPUT_NAMES,
    INFRASTRUCTURE_OUTPUT_NAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    build_trt_agent,
)
from skimage.draw import polygon
from einops import rearrange
from occ_input_trace import OccInputTrace, parse_frames


def patch_disk_backend_file_objects():
    original_get = HardDiskBackend.get

    def compatible_get(backend, filepath):
        if hasattr(filepath, "read"):
            return filepath.read()
        return original_get(backend, filepath)

    HardDiskBackend.get = compatible_get


def scalarize(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return scalarize(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, np.ndarray):
        return scalarize(value.tolist())
    if isinstance(value, dict):
        return {key: scalarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalarize(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "fps_from_mean": float(1000.0 / values.mean()),
    }


def iou_components(prediction, target):
    prediction = prediction.reshape(1, -1)
    target = target.reshape(1, -1)
    intersection = (prediction * target).sum(dim=1).cpu()
    union = (
        prediction.sum(dim=1) + target.sum(dim=1) - intersection.to(prediction)
    ).cpu()
    score = intersection / (union + 1.0e-13)
    return score, intersection, union


def map_metrics(outputs, ego_data):
    device = outputs["lane_pred"].device
    labels = agent_tensor(ego_data, "gt_lane_labels", device, torch.int64)
    masks = agent_tensor(ego_data, "gt_lane_masks", device, torch.int32)
    drivable_pred = outputs["drivable_pred"].to(torch.int32)
    lane_pred = outputs["lane_pred"].to(torch.int32)
    drivable_gt = masks[0, -1]
    lanes_pred = (lane_pred.sum(0) > 0).to(torch.int32)
    lanes_gt = (masks[0, :-1].sum(0) > 0).to(torch.int32)
    class_gt = [
        (masks[0][labels[0] == class_id].sum(0) > 0).to(torch.int32)
        for class_id in range(3)
    ]

    values = {}
    for name, prediction, target in (
        ("drivable", drivable_pred, drivable_gt),
        ("lanes", lanes_pred, lanes_gt),
        ("divider", lane_pred[0], class_gt[0]),
        ("crossing", lane_pred[1], class_gt[1]),
        ("contour", lane_pred[2], class_gt[2]),
    ):
        score, intersection, union = iou_components(prediction, target)
        values[name + "_iou"] = score
        values[name + "_intersection"] = intersection
        values[name + "_union"] = union
    return values, drivable_gt


def consecutive_instance_ground_truth(instance, ignore_index=255):
    result = torch.zeros_like(instance)
    next_id = 1
    for identifier in torch.unique(instance):
        if int(identifier.item()) in (0, ignore_index):
            continue
        result[instance == identifier] = next_id
        next_id += 1
    return result.long()


class PlanningPostprocessor:
    planning_steps = 10
    occ_n_future_only_occ = 4
    bev_h = 200
    bev_w = 200
    occ_filter_range = 5.0
    sigma = 1.0
    alpha_collision = 5.0

    collision_optimization = PlanningHeadSingleMode.collision_optimization

    @staticmethod
    def _is_feasible(x, y, feasible_area):
        feasible_count = 0
        for dx, dy in (
            (0, 0),
            (0, 1),
            (0, -1),
            (1, 0),
            (1, 1),
            (1, -1),
            (-1, 0),
            (-1, 1),
            (-1, -1),
        ):
            nx, ny = x + dx, y + dy
            if 0 <= nx < feasible_area.shape[1] and 0 <= ny < feasible_area.shape[0]:
                feasible_count += int(feasible_area[nx, ny] == 1)
            else:
                feasible_count += 1
        return feasible_count >= 2

    def drivable_optimization(self, initial_trajectory, feasible_area):
        original_trajectory = initial_trajectory.clone()
        trajectory = initial_trajectory[0]
        adjusted_trajectory = trajectory.clone()
        grid_size = feasible_area.shape[0]

        bev_trajectory = trajectory.clone()
        bev_trajectory[:, 0] = (-trajectory[:, 1] + 51.2) / 102.4 * 199
        bev_trajectory[:, 1] = (trajectory[:, 0] + 51.2) / 102.4 * 199
        bev_trajectory = torch.clamp(bev_trajectory, 0, 199)
        adjusted_bev = bev_trajectory.clone()

        invalid = []
        for index, point in enumerate(bev_trajectory):
            x, y = point.int().tolist()
            if not (
                0 <= x < grid_size
                and 0 <= y < grid_size
                and self._is_feasible(x, y, feasible_area)
            ):
                invalid.append(index)
        if not invalid:
            return original_trajectory
        if invalid == list(range(invalid[0], self.planning_steps)):
            for index in invalid:
                adjusted_bev[index] = adjusted_bev[index - 1]

        adjusted_trajectory[:, 0] = adjusted_bev[:, 1] * (102.4 / 200) - 51.2
        adjusted_trajectory[:, 1] = -adjusted_bev[:, 0] * (102.4 / 200) + 51.2
        return adjusted_trajectory.unsqueeze(0)

    def __call__(self, trajectory, occupancy, drivable):
        trajectory = self.collision_optimization(trajectory, occupancy)
        return self.drivable_optimization(trajectory, drivable)


class PlanningMetric(BasePlanningMetric):
    def __init__(self, n_future=10):
        super().__init__(n_future=n_future)
        self.add_state(
            "obj_out", default=torch.zeros(self.n_future), dist_reduce_fx="sum"
        )
        self.add_state(
            "obj_box_out", default=torch.zeros(self.n_future), dist_reduce_fx="sum"
        )

    def evaluate_single_coll(self, trajectory, segmentation):
        points = np.array([
            [-self.W / 2.0, self.H / 2.0 + 0.5],
            [self.W / 2.0, self.H / 2.0 + 0.5],
            [self.W / 2.0, -self.H / 2.0 + 0.5],
            [-self.W / 2.0, -self.H / 2.0 + 0.5],
        ])
        points = (points - self.bx.cpu().numpy()) / self.dx.cpu().numpy()
        points[:, [0, 1]] = points[:, [1, 0]]
        rows, columns = polygon(points[:, 1], points[:, 0])
        footprint = np.concatenate([rows[:, None], columns[:, None]], axis=-1)

        horizon = trajectory.shape[0]
        grid_trajectory = trajectory.view(horizon, 1, 2)
        grid_trajectory[:, :, [0, 1]] = grid_trajectory[:, :, [1, 0]]
        grid_trajectory = grid_trajectory / self.dx
        grid_trajectory = grid_trajectory.cpu().numpy() + footprint
        rows = np.clip(
            grid_trajectory[:, :, 0].astype(np.int32),
            0,
            self.bev_dimension[0] - 1,
        )
        columns = np.clip(
            grid_trajectory[:, :, 1].astype(np.int32),
            0,
            self.bev_dimension[1] - 1,
        )
        collision = np.zeros(horizon, dtype=np.bool_)
        for index in range(horizon):
            valid = (
                (rows[index] >= 0)
                & (rows[index] < self.bev_dimension[0])
                & (columns[index] >= 0)
                & (columns[index] < self.bev_dimension[1])
            )
            collision[index] = np.any(
                segmentation[index, rows[index][valid], columns[index][valid]]
                .cpu()
                .numpy()
            )
        return torch.from_numpy(collision).to(device=trajectory.device)

    def evaluate_coll(self, trajectories, gt_trajectories, segmentation):
        batch, horizon, _ = trajectories.shape
        trajectories = trajectories * trajectories.new_tensor([-1, 1])
        gt_trajectories = gt_trajectories * gt_trajectories.new_tensor([-1, 1])
        point_collision = torch.zeros(horizon, device=segmentation.device)
        box_collision = torch.zeros(horizon, device=segmentation.device)
        for batch_index in range(batch):
            gt_box_collision = self.evaluate_single_coll(
                gt_trajectories[batch_index], segmentation[batch_index]
            )
            x_values = trajectories[batch_index, :, 0]
            y_values = trajectories[batch_index, :, 1]
            y_indices = ((y_values - self.bx[0]) / self.dx[0]).long()
            x_indices = ((x_values - self.bx[1]) / self.dx[1]).long()
            valid = (
                (y_indices >= 0)
                & (y_indices < self.bev_dimension[0])
                & (x_indices >= 0)
                & (x_indices < self.bev_dimension[1])
                & ~gt_box_collision
            )
            time_indices = torch.arange(horizon, device=segmentation.device)
            point_collision[time_indices[valid]] += segmentation[
                batch_index,
                time_indices[valid],
                y_indices[valid],
                x_indices[valid],
            ].long()
            predicted_box_collision = self.evaluate_single_coll(
                trajectories[batch_index], segmentation[batch_index]
            )
            valid_box = ~gt_box_collision
            box_collision[time_indices[valid_box]] += predicted_box_collision[
                time_indices[valid_box]
            ].long()
        return point_collision, box_collision

    def update(
        self, trajectories, gt_trajectories, gt_mask, segmentation, drivable_gt
    ):
        assert trajectories.shape == gt_trajectories.shape
        trajectories[..., 0] = -trajectories[..., 0]
        gt_trajectories[..., 0] = -gt_trajectories[..., 0]
        l2_distance = self.compute_L2(trajectories, gt_trajectories, gt_mask)
        point_collision, box_collision = self.evaluate_coll(
            trajectories[:, :, :2], gt_trajectories[:, :, :2], segmentation
        )
        batch, horizon, _ = trajectories.shape
        undrivable = 1 - drivable_gt.unsqueeze(0).unsqueeze(0).repeat(
            batch, horizon, 1, 1
        )
        point_out, box_out = self.evaluate_coll(
            trajectories[:, :, :2], gt_trajectories[:, :, :2], undrivable
        )
        self.obj_col += point_collision
        self.obj_box_col += box_collision
        self.obj_out += point_out
        self.obj_box_out += box_out
        self.L2 += l2_distance.sum(dim=0)
        self.total += len(trajectories)

    def compute(self):
        return {
            "obj_col": self.obj_col / self.total,
            "obj_box_col": self.obj_box_col / self.total,
            "obj_out": self.obj_out / self.total,
            "obj_box_out": self.obj_box_out / self.total,
            "L2": self.L2 / self.total,
        }


class DeploymentPytorchEngine:
    def __init__(self, model, input_names, output_names):
        self.model = model.cuda().eval()
        self.input_names = list(input_names)
        self.output_names = list(output_names)

    def infer(self, inputs):
        values = [inputs[name] for name in self.input_names]
        with torch.no_grad():
            outputs = self.model.forward_uniad_trt(*values)
        if len(outputs) != len(self.output_names):
            raise RuntimeError((len(outputs), len(self.output_names)))
        return dict(zip(self.output_names, outputs))


def reference_occ_forward(head, x, ins_query):
    base_state = rearrange(
        x, "(h w) b d -> b d h w", h=head.bev_size[0]
    )
    if head.bevslicer:
        base_state = head.bev_sampler(base_state)
    base_state = head.bev_light_proj(base_state)
    base_state = head.base_downscale(base_state)
    last_state = base_state
    last_ins_query = ins_query
    future_states = []
    temporal_embeds = []
    layers_per_block = head.num_trans_layers // head.n_future_blocks
    for block_index in range(head.n_future_blocks):
        current_state = head.downscale_convs[block_index](last_state)
        current_query = head.temporal_mlps[block_index](last_ins_query)
        attention_mask, _, current_embed = head.get_attn_mask(
            current_state, current_query
        )
        current_state = rearrange(current_state, "b c h w -> (h w) b c")
        current_query = rearrange(current_query, "b q c -> q b c")
        for layer_offset in range(layers_per_block):
            layer_index = block_index * layers_per_block + layer_offset
            current_state = head.transformer_decoder.layers[layer_index](
                query=current_state,
                key=current_query,
                value=current_query,
                query_pos=None,
                key_pos=None,
                attn_masks=[None, attention_mask],
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
        current_state = rearrange(
            current_state,
            "(h w) b c -> b c h w",
            h=head.bev_size[0] // 8,
        )
        current_state = head.upsample_adds[block_index](
            current_state, last_state
        )
        future_states.append(current_state)
        temporal_embeds.append(current_embed)
        last_state = current_state
    future_states = head.dense_decoder(torch.stack(future_states, dim=1))
    if hasattr(head, "pred_prob"):
        probabilities = [
            head.pred_prob[index](future_states[:, index])
            for index in range(future_states.size(1))
        ]
        return torch.stack(probabilities, dim=1).permute(0, 2, 1, 3, 4)
    occupancy_query = head.query_to_occ_feat(torch.stack(temporal_embeds, dim=1))
    return torch.einsum(
        "btqc,btchw->bqthw", occupancy_query, future_states
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--fixed-track-count", type=int, default=960)
    parser.add_argument("--fixed-coop-count", type=int, default=64)
    parser.add_argument("--legacy-can-bus-deltas", action="store_true")
    parser.add_argument(
        "--agent-normalization-epsilon", type=float, default=0.0009765625
    )
    parser.add_argument("--occ-reference-forward", action="store_true")
    parser.add_argument("--occ-input-trace-dir")
    parser.add_argument("--occ-input-trace-frames", default="0,13,19")
    parser.add_argument(
        "--occ-input-trace-agent",
        choices=("all", "infrastructure", "ego"),
        default="all",
    )
    parser.add_argument("--occ-deep-encoder-trace", action="store_true")
    parser.add_argument("--occupancy-dump")
    parser.add_argument(
        "--frame-dump-dir",
        help="Read frame_*.pt files exported by dump_eval_frames.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.output_dir)
    patch_disk_backend_file_objects()

    cfg = Config.fromfile(args.config)
    cfg.data.workers_per_gpu = args.workers
    cfg.data.test.test_mode = True
    if args.frame_dump_dir:
        frame_paths = sorted(
            osp.join(args.frame_dump_dir, name)
            for name in os.listdir(args.frame_dump_dir)
            if name.startswith("frame_") and name.endswith(".pt")
        )
        if not frame_paths:
            raise FileNotFoundError("No frame_*.pt files in " + args.frame_dump_dir)
        dataset = None
        loader = (torch.load(path, map_location="cpu") for path in frame_paths)
        dataset_length = len(frame_paths)
        planning_steps = 10
    else:
        dataset = build_from_cfg(cfg.data.test, DATASETS)
        loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=args.workers,
            dist=False,
            shuffle=False,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )
        dataset_length = len(dataset)
        planning_steps = dataset.planning_steps

    _, infrastructure_model = build_trt_agent(
        args.config, args.checkpoint, "infrastructure"
    )
    if args.occ_reference_forward:
        infrastructure_model.occ_head.forward_trt = types.MethodType(
            reference_occ_forward, infrastructure_model.occ_head
        )
    infrastructure_engine = DeploymentPytorchEngine(
        infrastructure_model, INPUT_NAMES, INFRASTRUCTURE_OUTPUT_NAMES
    )
    _, ego_model = build_trt_agent(args.config, args.checkpoint, "ego")
    if args.occ_reference_forward:
        ego_model.occ_head.forward_trt = types.MethodType(
            reference_occ_forward, ego_model.occ_head
        )
    ego_model.cross_agent_query_interaction.normalization_epsilon = (
        args.agent_normalization_epsilon
    )
    occ_trace = None
    if args.occ_input_trace_dir:
        occ_trace = OccInputTrace(
            args.occ_input_trace_dir,
            parse_frames(args.occ_input_trace_frames),
        )
        if args.occ_input_trace_agent in ("all", "infrastructure"):
            occ_trace.wrap(
                infrastructure_model.occ_head, "infrastructure", "forward_trt"
            )
            occ_trace.wrap_common_encoder_stages(
                infrastructure_model,
                "infrastructure",
                deep=args.occ_deep_encoder_trace,
            )
            occ_trace.wrap_detection_stage(infrastructure_model, "infrastructure")
        if args.occ_input_trace_agent in ("all", "ego"):
            occ_trace.wrap(ego_model.occ_head, "ego", "forward_trt")
            occ_trace.wrap_common_encoder_stages(
                ego_model, "ego", deep=args.occ_deep_encoder_trace
            )
            occ_trace.wrap_cooperative_stages(ego_model, "ego")
            occ_trace.wrap_detection_stage(ego_model, "ego")
            occ_trace.wrap_motion_stage(ego_model, "ego")
    ego_engine = DeploymentPytorchEngine(ego_model, EGO_INPUT_NAMES, OUTPUT_NAMES)

    device = torch.device("cuda")
    infrastructure_state = AgentState(
        device, args.fixed_track_count, args.legacy_can_bus_deltas
    )
    ego_state = AgentState(device, args.fixed_track_count, args.legacy_can_bus_deltas)

    ranges = {"30x30": (70, 130), "100x100": (0, 200)}
    iou_metrics = {key: IntersectionOverUnion(2).cuda() for key in ranges}
    panoptic_metrics = {
        key: PanopticMetric(n_classes=2, temporally_consistent=True).cuda()
        for key in ranges
    }
    planning_metric = PlanningMetric(n_future=planning_steps).cuda()
    planning_postprocess = PlanningPostprocessor()

    frame_limit = min(args.max_frames, dataset_length)
    rows = []
    num_occ = 0
    occupancy_predictions = []
    occupancy_ground_truth = []
    occupancy_invalid = []
    progress = mmcv.ProgressBar(frame_limit)
    iterator = iter(loader)

    for frame_index in range(frame_limit):
        if occ_trace:
            occ_trace.set_frame(frame_index)
        end_to_end_start = time.perf_counter()
        data_start = end_to_end_start
        data = next(iterator)
        data_end = time.perf_counter()
        ego_data = data["ego_agent_data"]
        infrastructure_data = data["other_agent_data_dict"]["model_other_agent_inf"]

        torch.cuda.synchronize()
        infrastructure_start = time.perf_counter()
        infrastructure_inputs, _ = infrastructure_state.build_inputs(
            infrastructure_data
        )
        infrastructure_outputs = infrastructure_engine.infer(infrastructure_inputs)
        torch.cuda.synchronize()
        infrastructure_end = time.perf_counter()
        infrastructure_state.update(
            infrastructure_outputs,
            timestamp_fallback=infrastructure_inputs["timestamp"],
        )

        ego_inputs, _ = ego_state.build_inputs(ego_data)
        # UniV2X stores the vehicle-to-infrastructure calibration on the
        # infrastructure sample. The ego sample contains an identity matrix.
        veh2inf_rt = agent_tensor(
            infrastructure_data, "veh2inf_rt", device, torch.float32
        )
        cooperative_inputs, _ = prepare_cooperative_inputs(
            infrastructure_outputs,
            ego_state,
            veh2inf_rt,
            fixed_coop_count=args.fixed_coop_count,
        )
        ego_inputs.update(cooperative_inputs)
        torch.cuda.synchronize()
        ego_start = time.perf_counter()
        ego_outputs = ego_engine.infer(ego_inputs)
        torch.cuda.synchronize()
        ego_end = time.perf_counter()
        if occ_trace:
            occ_trace.flush()
        ego_state.update(ego_outputs, timestamp_fallback=ego_inputs["timestamp"])

        occupancy = ego_outputs["seg_out"].long()
        instance_prediction = predict_instance_segmentation_and_trajectories(
            occupancy, ego_outputs["occ_pred_sigmoid"]
        )
        instance_gt = consecutive_instance_ground_truth(
            agent_tensor(ego_data, "gt_instance", device, torch.int64)[:, :5]
        )
        segmentation_gt = agent_tensor(
            ego_data, "gt_segmentation", device, torch.int64
        )[:, :5, None]
        invalid_occ = bool(
            agent_tensor(
                ego_data, "gt_occ_has_invalid_frame", device, torch.bool
            ).item()
        )
        if args.occupancy_dump:
            occupancy_predictions.append(
                occupancy.detach().cpu().numpy().astype(np.uint8)
            )
            occupancy_ground_truth.append(
                segmentation_gt.detach().cpu().numpy().astype(np.uint8)
            )
            occupancy_invalid.append(invalid_occ)
        if not invalid_occ:
            num_occ += 1
            for key, (start, end) in ranges.items():
                limits = slice(start, end)
                iou_metrics[key](
                    occupancy[..., limits, limits].contiguous(),
                    segmentation_gt[..., limits, limits].contiguous(),
                )
                panoptic_metrics[key](
                    instance_prediction[..., limits, limits].contiguous(),
                    instance_gt[..., limits, limits].contiguous(),
                )

        _, drivable_gt = map_metrics(ego_outputs, ego_data)
        planning_trajectory = planning_postprocess(
            ego_outputs["outs_planning"], occupancy, drivable_gt
        )
        steps = planning_steps
        planning_metric(
            planning_trajectory[:, :steps, :2].clone(),
            agent_tensor(ego_data, "sdc_planning", device, torch.float32)[
                0, :, :steps, :2
            ].clone(),
            agent_tensor(ego_data, "sdc_planning_mask", device, torch.float32)[
                0, :, :steps, :2
            ].clone(),
            agent_tensor(ego_data, "gt_segmentation", device, torch.int64)[
                :, 1 : steps + 1
            ],
            drivable_gt,
        )
        end_to_end_end = time.perf_counter()
        rows.append({
            "frame": frame_index,
            "data_ms": (data_end - data_start) * 1000.0,
            "infrastructure_forward_ms": (
                infrastructure_end - infrastructure_start
            ) * 1000.0,
            "ego_forward_ms": (ego_end - ego_start) * 1000.0,
            "forward_ms": (ego_end - infrastructure_start) * 1000.0,
            "end_to_end_ms": (end_to_end_end - end_to_end_start) * 1000.0,
        })
        progress.update()

    occupancy_result = {}
    for key in ranges:
        panoptic = panoptic_metrics[key].compute()
        for metric_name, value in panoptic.items():
            occupancy_result.setdefault(metric_name, []).append(100 * value[1].item())
        occupancy_result.setdefault("iou", []).append(
            100 * iou_metrics[key].compute()[1].item()
        )
    occupancy_result["num_occ"] = num_occ
    occupancy_result["ratio_occ"] = num_occ / frame_limit
    metrics = {
        "occupancy": scalarize(occupancy_result),
        "planning": scalarize(planning_metric.compute()),
    }
    with open(osp.join(args.output_dir, "task_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)

    measured = rows[min(args.warmup_frames, max(0, len(rows) - 1)) :]
    latency = {
        "schema_version": 1,
        "framework": "PyTorch deployment graph",
        "precision": "FP32",
        "gpu": torch.cuda.get_device_name(0),
        "dataset_frames": frame_limit,
        "agent_normalization_epsilon": args.agent_normalization_epsilon,
        "infrastructure_forward": summarize(
            [row["infrastructure_forward_ms"] for row in measured]
        ),
        "ego_forward": summarize([row["ego_forward_ms"] for row in measured]),
        "forward": summarize([row["forward_ms"] for row in measured]),
        "end_to_end": summarize([row["end_to_end_ms"] for row in measured]),
    }
    with open(osp.join(args.output_dir, "latency_metrics.json"), "w") as handle:
        json.dump(latency, handle, indent=2, allow_nan=False)

    if args.occupancy_dump:
        mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.occupancy_dump)))
        np.savez_compressed(
            args.occupancy_dump,
            prediction=np.stack(occupancy_predictions),
            ground_truth=np.stack(occupancy_ground_truth),
            invalid=np.asarray(occupancy_invalid, dtype=np.bool_),
        )
    print(json.dumps({"task_metrics": metrics, "latency": latency}, indent=2))


if __name__ == "__main__":
    main()
