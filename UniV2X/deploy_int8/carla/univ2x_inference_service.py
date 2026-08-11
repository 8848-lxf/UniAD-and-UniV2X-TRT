#!/usr/bin/env python3
"""Persistent UniV2X TensorRT service for a Python 3.7 CARLA client."""

import argparse
import json
import os
import socket
import sys
import time

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from cooperative_runtime import prepare_cooperative_inputs
from trt_engine import TensorRTEngine
from eval_univ2x_carla_replay import (
    ReplayAgentState,
    filtered_inputs,
    lidar2image,
    load_templates,
    model_image,
    nonfinite,
    pose_inputs,
    right_handed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infrastructure_engine")
    parser.add_argument("ego_engine")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--template-input-dir", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--fixed-track-count", type=int, default=960)
    parser.add_argument("--fixed-coop-count", type=int, default=64)
    parser.add_argument("--context-cache-size", type=int, default=1)
    return parser.parse_args()


def atomic_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def finite_or_raise(side, outputs):
    bad = nonfinite(outputs)
    if bad:
        raise RuntimeError("non-finite %s outputs: %r" % (side, bad))


class UniV2XService:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        templates = load_templates(args.template_input_dir, self.device)
        self.infrastructure_engine = TensorRTEngine(
            args.infrastructure_engine,
            args.plugin,
            context_cache_size=args.context_cache_size,
        )
        self.ego_engine = TensorRTEngine(
            args.ego_engine,
            args.plugin,
            context_cache_size=args.context_cache_size,
        )
        self.templates = templates
        self.stream = torch.cuda.Stream()
        self.rows = []
        self.reset("carla_closed_loop")

    def reset(self, scene_token):
        self.infrastructure_state = ReplayAgentState(
            self.device,
            self.templates["infrastructure"],
            scene_token,
            self.args.fixed_track_count,
        )
        self.ego_state = ReplayAgentState(
            self.device,
            self.templates["ego"],
            scene_token,
            self.args.fixed_track_count,
        )
        return {"ok": True, "operation": "reset", "scene_token": scene_token}

    def infer(self, request):
        request_start = time.perf_counter()
        with np.load(request["npz"], allow_pickle=False) as arrays:
            camera = {
                "width": int(arrays["camera_width"][0]),
                "height": int(arrays["camera_height"][0]),
                "fov_degrees": float(arrays["camera_fov_degrees"][0]),
            }
            ego_image = model_image(arrays["ego_bgr"])
            infrastructure_image = model_image(arrays["infrastructure_bgr"])
            timestamp = float(arrays["timestamp"][0])
            ego_world = arrays["world_from_ego"]
            infrastructure_world = arrays["world_from_infrastructure"]
            ego_l2g_r, ego_l2g_t, ego_can_bus = pose_inputs(
                ego_world,
                arrays["ego_velocity"],
                arrays["ego_acceleration"],
                arrays["ego_angular_velocity"],
            )
            infra_l2g_r, infra_l2g_t, infra_can_bus = pose_inputs(
                infrastructure_world
            )
            ego_projection = lidar2image(
                ego_world, arrays["world_from_ego_camera"], camera
            )
            infrastructure_projection = lidar2image(
                infrastructure_world,
                arrays["world_from_infrastructure_camera"],
                camera,
            )

        with torch.cuda.stream(self.stream):
            infrastructure_inputs = self.infrastructure_state.build_inputs(
                infrastructure_image,
                timestamp,
                infra_l2g_r,
                infra_l2g_t,
                infra_can_bus,
                infrastructure_projection,
            )
            self.stream.synchronize()
            infrastructure_start = time.perf_counter()
            infrastructure_outputs = self.infrastructure_engine.infer(
                filtered_inputs(self.infrastructure_engine, infrastructure_inputs),
                synchronize=False,
            )
            self.stream.synchronize()
            infrastructure_end = time.perf_counter()
            finite_or_raise("infrastructure", infrastructure_outputs)
            self.infrastructure_state.update(infrastructure_outputs)

            ego_inputs = self.ego_state.build_inputs(
                ego_image,
                timestamp,
                ego_l2g_r,
                ego_l2g_t,
                ego_can_bus,
                ego_projection,
            )
            infra_from_ego = np.linalg.inv(
                right_handed(infrastructure_world)
            ).dot(right_handed(ego_world)).T.astype(np.float32)[None]
            cooperative, cooperative_stats = prepare_cooperative_inputs(
                infrastructure_outputs,
                self.ego_state,
                torch.from_numpy(infra_from_ego).to(self.device),
                fixed_coop_count=self.args.fixed_coop_count,
            )
            ego_inputs.update(cooperative)
            self.stream.synchronize()
            ego_start = time.perf_counter()
            ego_outputs = self.ego_engine.infer(
                filtered_inputs(self.ego_engine, ego_inputs), synchronize=False
            )
            self.stream.synchronize()
            ego_end = time.perf_counter()
            finite_or_raise("ego", ego_outputs)
            self.ego_state.update(ego_outputs)

        planning = ego_outputs["outs_planning"].detach().float().cpu().numpy()[0]
        scores = ego_outputs["det_scores"].detach().float().cpu().numpy()
        response = {
            "ok": True,
            "frame": int(request["frame"]),
            "planning_xy": planning[:, :2].tolist(),
            "detections_above_0_1": int((scores >= 0.1).sum()),
            "cooperative": cooperative_stats,
            "latency_ms": {
                "infrastructure_forward": (infrastructure_end - infrastructure_start) * 1000.0,
                "ego_forward": (ego_end - ego_start) * 1000.0,
                "combined_forward": (ego_end - infrastructure_start) * 1000.0,
                "service_end_to_end": (time.perf_counter() - request_start) * 1000.0,
            },
        }
        self.rows.append(response)
        atomic_json(
            self.args.output,
            {
                "schema": "univ2x_carla_closed_loop_inference_service_v1",
                "precision": self.args.precision,
                "frames": len(self.rows),
                "fixed_track_count": self.args.fixed_track_count,
                "fixed_coop_count": self.args.fixed_coop_count,
                "rows": self.rows,
            },
        )
        return response


def serve(args):
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    os.makedirs(os.path.dirname(os.path.abspath(args.socket)), exist_ok=True)
    service = UniV2XService(args)
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
