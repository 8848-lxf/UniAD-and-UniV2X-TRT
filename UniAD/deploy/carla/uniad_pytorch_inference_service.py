#!/usr/bin/env python3
"""Persistent trained-checkpoint UniAD service for the CARLA client."""

import argparse
import importlib
import json
import os
import socket
import sys
import time

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY_ROOT = os.environ.get("UNIAD_DEPLOY_ROOT") or os.path.abspath(
    os.path.join(HERE, "..", "..", "repro", "UniAD_deploy")
)
if DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, DEPLOY_ROOT)

from mmcv import Config
from mmcv.runner import load_checkpoint
from third_party.uniad_mmdet3d.models.builder import build_model

from uniad_inference_service import (
    UniADState,
    atomic_json,
    lidar2image,
    model_images,
    nonfinite,
    pose_inputs,
)
from planning_postprocess import postprocess_planning


INPUT_NAMES = [
    *["prev_track_intances%d" % index for index in range(14)],
    "prev_timestamp", "prev_l2g_r_mat", "prev_l2g_t", "prev_bev",
    "gt_lane_labels", "gt_lane_masks", "gt_segmentation",
    "img_metas_scene_token", "timestamp", "l2g_r_mat", "l2g_t", "img",
    "img_metas_can_bus", "img_metas_lidar2img", "image_shape", "command",
    "use_prev_bev", "max_obj_id",
]
TRACK_OUTPUT_NAMES = [
    "prev_track_intances%d_out" % index
    for index in (0, 1, 3, 4, 5, 6, 8, 9, 11, 12, 13)
]
OUTPUT_NAMES = [
    *TRACK_OUTPUT_NAMES,
    "prev_timestamp_out", "prev_l2g_t_out", "prev_l2g_r_mat_out", "bev_embed",
    "bboxes_dict_bboxes", "scores", "labels", "bbox_index", "obj_idxes",
    "max_obj_id_out", "seg_out", "outs_planning",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-track-count", type=int, default=1300)
    return parser.parse_args()


def load_model(config_path, checkpoint_path):
    os.chdir(DEPLOY_ROOT)
    cfg = Config.fromfile(config_path)
    if cfg.get("plugin", False):
        plugin_dir = cfg.get("plugin_dir") or os.path.dirname(config_path)
        module = ".".join(os.path.normpath(plugin_dir).strip(os.sep).split(os.sep))
        if os.path.isabs(plugin_dir):
            module = ".".join(os.path.relpath(plugin_dir, DEPLOY_ROOT).split(os.sep))
        importlib.import_module(module.rstrip("."))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    return model.cuda().eval()


class UniADPyTorchService:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.model = load_model(args.config, args.checkpoint)
        self.stream = torch.cuda.Stream()
        self.state = UniADState(self.device, args.fixed_track_count)
        self.rows = []

    def reset(self, scene_token):
        self.state.reset()
        return {"ok": True, "operation": "reset", "scene_token": scene_token}

    def _model_inputs(self, values):
        rows = self.args.fixed_track_count
        values.update({
            "prev_track_intances2": torch.zeros(rows, 256, device=self.device),
            "prev_track_intances7": torch.zeros(rows, device=self.device),
            "prev_track_intances10": torch.zeros(rows, 10, device=self.device),
            "gt_lane_labels": torch.zeros(
                1, 1, dtype=torch.float32, device=self.device
            ),
            "gt_lane_masks": torch.zeros(
                1, 1, 50, 50, dtype=torch.float32, device=self.device
            ),
            "gt_segmentation": torch.zeros(
                1, 7, 50, 50, dtype=torch.float32, device=self.device
            ),
            "img_metas_scene_token": torch.zeros(
                32, dtype=torch.uint8, device=self.device
            ),
            "image_shape": torch.tensor(
                [256.0, 416.0], dtype=torch.float32, device=self.device
            ),
        })
        return tuple(values[name] for name in INPUT_NAMES)

    def infer(self, request):
        request_start = time.perf_counter()
        with np.load(request["npz"], allow_pickle=False) as arrays:
            width = int(arrays["camera_width"][0])
            height = int(arrays["camera_height"][0])
            fov = float(arrays["camera_fov_degrees"][0])
            timestamp = float(arrays["timestamp"][0])
            l2g_r, l2g_t, can_bus = pose_inputs(
                arrays["world_from_ego"], arrays["ego_velocity"],
                arrays["ego_acceleration"], arrays["ego_angular_velocity"],
            )
            projections = np.stack([
                lidar2image(arrays["world_from_ego"], camera, width, height, fov)
                for camera in arrays["world_from_cameras"]
            ])
            command = int(arrays["command"][0])
            camera_bgr = arrays["camera_bgr"]

        with torch.cuda.stream(self.stream), torch.no_grad():
            images = model_images(camera_bgr, self.device)
            values = self.state.build_inputs(
                images, timestamp, l2g_r, l2g_t, can_bus, projections, command
            )
            inputs = self._model_inputs(values)
            self.stream.synchronize()
            forward_start = time.perf_counter()
            result = self.model.forward_uniad_trt(*inputs)
            self.stream.synchronize()
            forward_end = time.perf_counter()
            if len(result) != len(OUTPUT_NAMES):
                raise RuntimeError("unexpected UniAD output count: %d" % len(result))
            outputs = dict(zip(OUTPUT_NAMES, result))
            bad = nonfinite(outputs)
            if bad:
                raise RuntimeError("non-finite UniAD outputs: %r" % bad)
            self.state.update(outputs)

        postprocess_start = time.perf_counter()
        raw_planning = outputs["outs_planning"].detach().float().cpu().numpy()[0]
        planning, postprocess_statistics = postprocess_planning(
            raw_planning,
            outputs["seg_out"].detach().cpu().numpy(),
        )
        postprocess_end = time.perf_counter()
        scores = outputs["scores"].detach().float().cpu().numpy()
        response = {
            "ok": True,
            "frame": int(request["frame"]),
            "planning_xy": planning.tolist(),
            "planning_postprocess": postprocess_statistics,
            "detections_above_0_25": int((scores >= 0.25).sum()),
            "track_query_count": int(outputs["prev_track_intances3_out"].shape[0]),
            "latency_ms": {
                "engine_forward": (forward_end - forward_start) * 1000.0,
                "planning_postprocess": (
                    postprocess_end - postprocess_start
                ) * 1000.0,
                "service_end_to_end": (time.perf_counter() - request_start) * 1000.0,
            },
        }
        self.rows.append(response)
        atomic_json(self.args.output, {
            "schema": "uniad_carla_pytorch_inference_service_v1",
            "precision": "pytorch_fp32",
            "checkpoint": os.path.abspath(self.args.checkpoint),
            "frames": len(self.rows),
            "fixed_track_count": self.args.fixed_track_count,
            "planning_protocol": "occupancy-aware collision optimized",
            "rows": self.rows,
        })
        return response


def serve(args):
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    service = UniADPyTorchService(args)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(args.socket)
    listener.listen(1)
    print("READY %s" % args.socket, flush=True)
    try:
        connection, _ = listener.accept()
        with connection, connection.makefile("rwb") as channel:
            for raw in channel:
                request = {}
                try:
                    request = json.loads(raw.decode("utf-8"))
                    operation = request.get("op", "infer")
                    if operation == "infer":
                        response = service.infer(request)
                    elif operation == "reset":
                        response = service.reset(
                            request.get("scene_token", "carla_closed_loop")
                        )
                    elif operation == "stop":
                        response = {"ok": True, "operation": "stop"}
                    else:
                        raise ValueError("unknown operation: %s" % operation)
                except Exception as error:
                    response = {"ok": False, "error": repr(error)}
                channel.write((json.dumps(response, allow_nan=False) + "\n").encode("utf-8"))
                channel.flush()
                if request.get("op") == "stop":
                    break
    finally:
        listener.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    serve(parse_args())
