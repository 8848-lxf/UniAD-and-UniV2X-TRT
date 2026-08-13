#!/usr/bin/env python3
"""Persistent UniAD TensorRT service for the isolated CARLA client."""

import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as functional


HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "UniV2X", "deploy_int8", "tools"
))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from planning_postprocess import postprocess_planning

TRACK_SPECS = (
    ("prev_track_intances0", (512,), torch.float32),
    ("prev_track_intances1", (3,), torch.float32),
    ("prev_track_intances3", (), torch.int32),
    ("prev_track_intances4", (), torch.int32),
    ("prev_track_intances5", (), torch.int32),
    ("prev_track_intances6", (), torch.float32),
    ("prev_track_intances8", (), torch.float32),
    ("prev_track_intances9", (10,), torch.float32),
    ("prev_track_intances11", (4, 256), torch.float32),
    ("prev_track_intances12", (4,), torch.int32),
    ("prev_track_intances13", (), torch.float32),
)
RECURRENT_OUTPUT_NAMES = {
    *(name + "_out" for name, _, _ in TRACK_SPECS),
    "prev_timestamp_out",
    "prev_l2g_r_mat_out",
    "prev_l2g_t_out",
    "bev_embed",
}
HAND_FLIP = np.diag([1.0, -1.0, 1.0, 1.0])
CAMERA_RH_TO_STANDARD = np.asarray([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
RGB_MEAN = torch.tensor([123.675, 116.280, 103.530]).view(1, 1, 3, 1, 1)
RGB_STD = torch.tensor([58.395, 57.120, 57.375]).view(1, 1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--fixed-track-count", type=int, required=True)
    parser.add_argument("--context-cache-size", type=int, default=1)
    return parser.parse_args()


def atomic_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def right_handed(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return HAND_FLIP.dot(matrix).dot(HAND_FLIP)


def intrinsic(width, height, fov_degrees):
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    value = np.eye(4, dtype=np.float64)
    value[0, 0] = focal
    value[1, 1] = focal
    value[0, 2] = width / 2.0
    value[1, 2] = height / 2.0
    return value


def lidar2image(world_from_ego, world_from_camera, width, height, fov_degrees):
    world_from_ego = right_handed(world_from_ego)
    world_from_camera = right_handed(world_from_camera)
    camera_from_ego = np.linalg.inv(world_from_camera).dot(world_from_ego)
    return intrinsic(width, height, fov_degrees).dot(
        CAMERA_RH_TO_STANDARD
    ).dot(camera_from_ego).astype(np.float32)


def pose_inputs(world_from_ego, velocity, acceleration, angular_velocity):
    transform = right_handed(world_from_ego)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[:3] = translation
    can_bus[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    can_bus[7:10] = [acceleration[0], -acceleration[1], acceleration[2]]
    can_bus[10:13] = [angular_velocity[0], -angular_velocity[1], angular_velocity[2]]
    can_bus[13:16] = [velocity[0], -velocity[1], velocity[2]]
    can_bus[-2] = yaw
    can_bus[-1] = math.degrees(yaw) % 360.0
    # UniAD stores local-to-global rotations for row-vector multiplication.
    return rotation.T.astype(np.float32)[None], translation.astype(np.float32)[None], can_bus


def model_images(camera_bgr, device):
    if camera_bgr.shape != (6, 900, 1600, 3):
        raise ValueError("unexpected six-camera tensor shape: %r" % (camera_bgr.shape,))
    rgb = np.ascontiguousarray(camera_bgr[..., ::-1])
    value = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    value = value.permute(0, 3, 1, 2).unsqueeze(0)
    mean = RGB_MEAN.to(device)
    std = RGB_STD.to(device)
    value = (value - mean) / std
    value = functional.interpolate(
        value.reshape(6, 3, 900, 1600),
        size=(225, 400), mode="bilinear", align_corners=False,
    ).reshape(1, 6, 3, 225, 400)
    return functional.pad(value, (0, 16, 0, 31)).contiguous()


def nonfinite(outputs):
    return {
        name: int((~torch.isfinite(value)).sum().item())
        for name, value in outputs.items()
        if value.is_floating_point() and not torch.isfinite(value).all()
    }


class UniADState:
    def __init__(self, device, fixed_track_count):
        self.device = device
        self.fixed_track_count = fixed_track_count
        self.reset()

    def reset(self):
        self.tracks = {
            name: torch.zeros((901,) + tail, dtype=dtype, device=self.device)
            for name, tail, dtype in TRACK_SPECS
        }
        self.prev_timestamp = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.prev_l2g_r_mat = torch.zeros(1, 3, 3, dtype=torch.float32, device=self.device)
        self.prev_l2g_t = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.prev_bev = torch.zeros(2500, 1, 256, dtype=torch.float32, device=self.device)
        self.max_obj_id = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.timestamp_origin = None
        self.prev_position = None
        self.prev_angle = None
        self.frame_index = 0

    def padded_tracks(self):
        result = {}
        for name, tail, dtype in TRACK_SPECS:
            value = self.tracks[name]
            rows = int(value.shape[0])
            if rows > self.fixed_track_count:
                raise RuntimeError(
                    "%s row count %d exceeds fixed capacity %d"
                    % (name, rows, self.fixed_track_count)
                )
            if rows < self.fixed_track_count:
                padding = torch.full(
                    (self.fixed_track_count - rows,) + tail,
                    -10000, dtype=dtype, device=self.device,
                )
                value = torch.cat((value, padding), dim=0)
            result[name] = value
        return result

    def build_inputs(self, images, timestamp, l2g_r, l2g_t, can_bus, lidar2img, command):
        if self.timestamp_origin is None:
            self.timestamp_origin = timestamp
        relative_timestamp = timestamp - self.timestamp_origin
        relative_can_bus = can_bus.copy()
        if self.prev_position is None:
            relative_can_bus[:3] = 0.0
            relative_can_bus[-1] = 0.0
        else:
            relative_can_bus[:3] -= self.prev_position
            relative_can_bus[-1] -= self.prev_angle
        self.prev_position = can_bus[:3].copy()
        self.prev_angle = float(can_bus[-1])

        values = self.padded_tracks()
        values.update({
            "prev_timestamp": self.prev_timestamp,
            "prev_l2g_r_mat": self.prev_l2g_r_mat,
            "prev_l2g_t": self.prev_l2g_t,
            "prev_bev": self.prev_bev,
            "timestamp": torch.tensor([relative_timestamp], dtype=torch.float32, device=self.device),
            "l2g_r_mat": torch.from_numpy(l2g_r).to(self.device),
            "l2g_t": torch.from_numpy(l2g_t).to(self.device),
            "img": images,
            "img_metas_can_bus": torch.from_numpy(relative_can_bus).to(self.device),
            "img_metas_lidar2img": torch.from_numpy(lidar2img[None]).to(self.device),
            "command": torch.tensor([command], dtype=torch.float32, device=self.device),
            "use_prev_bev": torch.tensor(
                [int(self.frame_index > 0)], dtype=torch.int32, device=self.device
            ),
            "max_obj_id": self.max_obj_id,
        })
        self.frame_index += 1
        return values

    def update(self, outputs):
        for name, _, _ in TRACK_SPECS:
            self.tracks[name] = outputs[name + "_out"]
        self.prev_timestamp = outputs["prev_timestamp_out"]
        self.prev_l2g_r_mat = outputs["prev_l2g_r_mat_out"]
        self.prev_l2g_t = outputs["prev_l2g_t_out"]
        self.prev_bev = outputs["bev_embed"]
        self.max_obj_id = outputs["max_obj_id_out"]


class UniADService:
    def __init__(self, args):
        from trt_engine import TensorRTEngine

        self.args = args
        self.device = torch.device("cuda")
        self.engine = TensorRTEngine(
            args.engine, args.plugin, context_cache_size=args.context_cache_size
        )
        self.stream = torch.cuda.Stream()
        self.state = UniADState(self.device, args.fixed_track_count)
        self.rows = []
        self.temporal_recoveries = []

    def reset(self, scene_token):
        self.state.reset()
        return {"ok": True, "operation": "reset", "scene_token": scene_token}

    def infer(self, request):
        request_start = time.perf_counter()
        with np.load(request["npz"], allow_pickle=False) as arrays:
            camera_bgr = arrays["camera_bgr"]
            world_from_ego = arrays["world_from_ego"]
            camera_transforms = arrays["world_from_cameras"]
            width = int(arrays["camera_width"][0])
            height = int(arrays["camera_height"][0])
            fov = float(arrays["camera_fov_degrees"][0])
            timestamp = float(arrays["timestamp"][0])
            l2g_r, l2g_t, can_bus = pose_inputs(
                world_from_ego,
                arrays["ego_velocity"],
                arrays["ego_acceleration"],
                arrays["ego_angular_velocity"],
            )
            projections = np.stack([
                lidar2image(world_from_ego, camera, width, height, fov)
                for camera in camera_transforms
            ])
            command = int(arrays["command"][0])

        with torch.cuda.stream(self.stream):
            images = model_images(camera_bgr, self.device)
            inputs = self.state.build_inputs(
                images, timestamp, l2g_r, l2g_t, can_bus, projections, command
            )
            self.stream.synchronize()
            forward_start = time.perf_counter()
            outputs = self.engine.infer(inputs, synchronize=False)
            self.stream.synchronize()
            forward_end = time.perf_counter()
            bad = nonfinite(outputs)
            critical_bad = {
                name: count for name, count in bad.items()
                if name not in RECURRENT_OUTPUT_NAMES
            }
            if critical_bad:
                raise RuntimeError("non-finite UniAD outputs: %r" % bad)
            temporal_recovery = None
            if bad:
                # A CARLA route is much longer than one nuScenes scene. If only
                # recurrent state is invalid, keep the finite current outputs
                # and start a clean temporal segment on the next model call.
                temporal_recovery = {
                    "frame": int(request["frame"]),
                    "nonfinite_recurrent_outputs": bad,
                    "action": "discard_recurrent_outputs_and_reset_state",
                }
                self.temporal_recoveries.append(temporal_recovery)
                self.state.reset()
            else:
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
            "temporal_state_recovery": temporal_recovery,
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
            "schema": "uniad_carla_closed_loop_inference_service_v1",
            "precision": self.args.precision,
            "frames": len(self.rows),
            "fixed_track_count": self.args.fixed_track_count,
            "planning_protocol": "occupancy-aware collision optimized",
            "temporal_state_recoveries": self.temporal_recoveries,
            "rows": self.rows,
        })
        return response


def serve(args):
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    os.makedirs(os.path.dirname(os.path.abspath(args.socket)), exist_ok=True)
    service = UniADService(args)
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
                        response = service.reset(request.get("scene_token", "carla_closed_loop"))
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
