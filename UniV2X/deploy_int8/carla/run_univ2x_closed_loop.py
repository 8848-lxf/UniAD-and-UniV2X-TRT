#!/usr/bin/env python3
"""Run a model-controlled UniV2X closed loop on a CARLA route."""

from __future__ import print_function

import argparse
import json
import math
import os
import queue
import random
import socket
import time
import xml.etree.ElementTree as ET

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=40000)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--route-id", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--inference-interval", type=int, default=10)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--traffic-vehicles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--target-index", type=int, default=3)
    parser.add_argument("--steer-gain", type=float, default=1.25)
    parser.add_argument("--speed-gain", type=float, default=0.35)
    parser.add_argument("--brake-gain", type=float, default=0.5)
    return parser.parse_args()


def atomic_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def route_spec(path, requested_id):
    routes = ET.parse(path).getroot().findall("route")
    if requested_id is not None:
        routes = [route for route in routes if route.get("id") == str(requested_id)]
    if not routes:
        raise RuntimeError("route id not found in %s" % path)
    route = routes[0]
    points = []
    for waypoint in route.findall("waypoint"):
        points.append({name: float(waypoint.get(name, "0")) for name in (
            "x", "y", "z", "pitch", "yaw", "roll"
        )})
    if len(points) < 2:
        raise RuntimeError("route requires at least two waypoints")
    return route.get("town"), route.get("id"), points


def transform_matrix(transform):
    location = transform.location
    rotation = transform.rotation
    cy = math.cos(math.radians(rotation.yaw))
    sy = math.sin(math.radians(rotation.yaw))
    cp = math.cos(math.radians(rotation.pitch))
    sp = math.sin(math.radians(rotation.pitch))
    cr = math.cos(math.radians(rotation.roll))
    sr = math.sin(math.radians(rotation.roll))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ])
    matrix[:3, 3] = [location.x, location.y, location.z]
    return matrix


def camera_array(image):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4
    )
    return np.ascontiguousarray(bgra[:, :, :3])


def matching_measurement(sensor_queue, frame, name):
    deadline = time.time() + 20.0
    while time.time() < deadline:
        measurement = sensor_queue.get(timeout=max(0.1, deadline - time.time()))
        if measurement.frame == frame:
            return measurement
        if measurement.frame > frame:
            raise RuntimeError(
                "%s skipped requested frame %d and returned %d"
                % (name, frame, measurement.frame)
            )
    raise RuntimeError("timed out waiting for %s frame %d" % (name, frame))


def service_request(channel, value):
    channel.write((json.dumps(value, allow_nan=False) + "\n").encode("utf-8"))
    channel.flush()
    response = json.loads(channel.readline().decode("utf-8"))
    if not response.get("ok"):
        raise RuntimeError("inference service failed: %s" % response.get("error"))
    return response


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"samples": 0, "mean_ms": None, "p50_ms": None, "p99_ms": None}
    return {
        "samples": int(len(values)),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def spawn_traffic(world, traffic_manager, traffic_manager_port, count, seed):
    randomizer = random.Random(seed)
    points = list(world.get_map().get_spawn_points())
    randomizer.shuffle(points)
    blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    actors = []
    for point in points:
        if len(actors) >= count:
            break
        actor = world.try_spawn_actor(randomizer.choice(blueprints), point)
        if actor is not None:
            actor.set_autopilot(True, traffic_manager_port)
            actors.append(actor)
    return actors


def progress_percent(start, destination, current):
    route = np.asarray([destination.x - start.x, destination.y - start.y])
    travelled = np.asarray([current.x - start.x, current.y - start.y])
    denominator = float(np.dot(route, route))
    return float(np.clip(np.dot(travelled, route) / max(denominator, 1e-6), 0.0, 1.0) * 100.0)


def model_control(carla, planning, speed_mps, args):
    trajectory = np.asarray(planning, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2 or not np.isfinite(trajectory).all():
        raise RuntimeError("invalid planning trajectory")
    target_index = min(max(0, args.target_index), len(trajectory) - 1)
    target = trajectory[target_index, :2]
    distance = float(np.linalg.norm(target))
    horizon_seconds = 0.5 * float(target_index + 1)
    target_speed = float(np.clip(distance / max(horizon_seconds, 0.5), 1.5, 8.0))
    heading = math.atan2(float(target[1]), max(0.25, float(target[0])))
    control = carla.VehicleControl()
    # UniV2X uses x-forward/y-left; CARLA positive steering is right.
    control.steer = float(np.clip(-args.steer_gain * heading, -1.0, 1.0))
    speed_error = target_speed - speed_mps
    if distance < 0.25 or speed_error < -0.5:
        control.throttle = 0.0
        control.brake = float(np.clip(-args.brake_gain * speed_error, 0.0, 1.0))
    else:
        control.throttle = float(np.clip(args.speed_gain * speed_error, 0.0, 0.75))
        control.brake = 0.0
    return control, target_speed


def main():
    args = parse_args()
    os.makedirs(args.request_dir, exist_ok=True)
    town, route_id, points = route_spec(args.route, args.route_id)

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if not world.get_map().name.endswith(town):
        world = client.load_world(town)
    original_settings = world.get_settings()
    traffic_manager_port = args.port + 5
    traffic_manager = client.get_trafficmanager(traffic_manager_port)
    actors = []
    sensors = []
    rows = []
    events = {"collision": 0, "lane_invasion": 0}
    completion_reason = "max_frames"

    service_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    service_socket.settimeout(120.0)
    service_socket.connect(args.socket)
    channel = service_socket.makefile("rwb")

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        start = points[0]
        destination = points[-1]
        requested_start_transform = carla.Transform(
            carla.Location(x=start["x"], y=start["y"], z=start["z"] + 0.5),
            carla.Rotation(pitch=start["pitch"], yaw=start["yaw"], roll=start["roll"]),
        )
        destination_location = carla.Location(
            x=destination["x"], y=destination["y"], z=destination["z"]
        )
        blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name", "hero")
        road_start = world.get_map().get_waypoint(
            requested_start_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        candidate_transforms = []
        if road_start is not None:
            candidate_transforms.append(road_start.transform)
            candidate_transforms.extend(
                waypoint.transform for waypoint in road_start.previous(2.0)
            )
            candidate_transforms.extend(
                waypoint.transform for waypoint in road_start.next(2.0)
            )
        candidate_transforms.append(requested_start_transform)
        ego = None
        start_transform = None
        for candidate in candidate_transforms:
            for height_offset in (0.5, 1.5, 2.5):
                # Keep the route direction from the XML. The projected road
                # waypoint can belong to the opposite-direction lane.
                attempted = carla.Transform(
                    carla.Location(
                        x=candidate.location.x,
                        y=candidate.location.y,
                        z=candidate.location.z + height_offset,
                    ),
                    carla.Rotation(
                        pitch=requested_start_transform.rotation.pitch,
                        yaw=requested_start_transform.rotation.yaw,
                        roll=requested_start_transform.rotation.roll,
                    ),
                )
                ego = world.try_spawn_actor(blueprint, attempted)
                if ego is not None:
                    start_transform = attempted
                    break
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError("no collision-free ego spawn near route start")
        actors.append(ego)
        actors.extend(spawn_traffic(
            world, traffic_manager, traffic_manager_port,
            args.traffic_vehicles, args.seed + 1,
        ))

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1920")
        camera_bp.set_attribute("image_size_y", "1080")
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", "0.0")
        ego_camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=1.8)),
            attach_to=ego,
        )
        sensors.append(ego_camera)

        waypoint = world.get_map().get_waypoint(start_transform.location)
        ahead = waypoint.next(25.0)
        roadside = ahead[0] if ahead else waypoint
        yaw = math.radians(roadside.transform.rotation.yaw)
        infrastructure_transform = carla.Transform(
            carla.Location(
                x=roadside.transform.location.x - 5.0 * math.sin(yaw),
                y=roadside.transform.location.y + 5.0 * math.cos(yaw),
                z=roadside.transform.location.z + 5.5,
            ),
            carla.Rotation(
                pitch=-18.0,
                yaw=roadside.transform.rotation.yaw + 180.0,
                roll=0.0,
            ),
        )
        infrastructure_camera = world.spawn_actor(camera_bp, infrastructure_transform)
        sensors.append(infrastructure_camera)

        collision_sensor = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.collision"),
            carla.Transform(),
            attach_to=ego,
        )
        collision_sensor.listen(
            lambda event: events.__setitem__("collision", events["collision"] + 1)
        )
        sensors.append(collision_sensor)
        lane_sensor = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=ego,
        )
        lane_sensor.listen(
            lambda event: events.__setitem__("lane_invasion", events["lane_invasion"] + 1)
        )
        sensors.append(lane_sensor)

        ego_queue = queue.Queue()
        infrastructure_queue = queue.Queue()
        ego_camera.listen(ego_queue.put)
        infrastructure_camera.listen(infrastructure_queue.put)
        service_request(channel, {
            "op": "reset",
            "scene_token": "carla_%s_route_%s" % (town, route_id),
        })

        control = carla.VehicleControl(throttle=0.0, brake=1.0)
        for _ in range(10):
            ego.apply_control(control)
            world.tick()

        request_path = os.path.join(args.request_dir, "current_frame.npz")
        for frame_index in range(args.max_frames):
            tick_start = time.perf_counter()
            ego.apply_control(control)
            carla_frame = world.tick()
            snapshot = world.get_snapshot()
            location = ego.get_location()
            velocity = ego.get_velocity()
            speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            distance_to_destination = location.distance(destination_location)
            inference = None
            target_speed = None

            if frame_index % args.inference_interval == 0:
                ego_image = matching_measurement(ego_queue, carla_frame, "ego camera")
                infrastructure_image = matching_measurement(
                    infrastructure_queue, carla_frame, "infrastructure camera"
                )
                ego_transform = ego.get_transform()
                acceleration = ego.get_acceleration()
                angular_velocity = ego.get_angular_velocity()
                np.savez(
                    request_path,
                    ego_bgr=camera_array(ego_image),
                    infrastructure_bgr=camera_array(infrastructure_image),
                    world_from_ego=transform_matrix(ego_transform),
                    world_from_ego_camera=transform_matrix(ego_camera.get_transform()),
                    world_from_infrastructure=transform_matrix(infrastructure_transform),
                    world_from_infrastructure_camera=transform_matrix(
                        infrastructure_camera.get_transform()
                    ),
                    timestamp=np.asarray([snapshot.timestamp.elapsed_seconds], dtype=np.float64),
                    ego_velocity=np.asarray([velocity.x, velocity.y, velocity.z], dtype=np.float32),
                    ego_acceleration=np.asarray(
                        [acceleration.x, acceleration.y, acceleration.z], dtype=np.float32
                    ),
                    ego_angular_velocity=np.asarray(
                        [angular_velocity.x, angular_velocity.y, angular_velocity.z],
                        dtype=np.float32,
                    ),
                    camera_width=np.asarray([1920], dtype=np.int32),
                    camera_height=np.asarray([1080], dtype=np.int32),
                    camera_fov_degrees=np.asarray([90.0], dtype=np.float32),
                )
                inference = service_request(channel, {
                    "op": "infer", "frame": frame_index, "npz": request_path
                })
                control, target_speed = model_control(
                    carla, inference["planning_xy"], speed_mps, args
                )

            row = {
                "frame": frame_index,
                "carla_frame": int(carla_frame),
                "sim_time_seconds": float(snapshot.timestamp.elapsed_seconds),
                "speed_mps": speed_mps,
                "target_speed_mps": target_speed,
                "distance_to_destination_m": float(distance_to_destination),
                "route_progress_percent": progress_percent(
                    start_transform.location, destination_location, location
                ),
                "control": {
                    "throttle": float(control.throttle),
                    "steer": float(control.steer),
                    "brake": float(control.brake),
                },
                "events_cumulative": dict(events),
                "wall_tick_ms": (time.perf_counter() - tick_start) * 1000.0,
                "inference": inference,
            }
            rows.append(row)
            if frame_index % 10 == 0:
                print(
                    "frame=%d progress=%.2f%% distance=%.2fm speed=%.2fm/s"
                    % (frame_index, row["route_progress_percent"], distance_to_destination, speed_mps),
                    flush=True,
                )
                atomic_json(args.output + ".progress", {
                    "frame": frame_index,
                    "route_progress_percent": row["route_progress_percent"],
                    "distance_to_destination_m": distance_to_destination,
                    "events": events,
                })
            if distance_to_destination <= 5.0:
                completion_reason = "destination_reached"
                break

        inference_rows = [row["inference"] for row in rows if row["inference"]]
        result = {
            "schema": "univ2x_carla_model_controlled_closed_loop_v1",
            "scope": (
                "Model-controlled CARLA closed loop using synchronized ego and "
                "roadside RGB cameras; custom route protocol, not an official "
                "V2Xverse leaderboard score."
            ),
            "route": {
                "file": os.path.abspath(args.route),
                "id": route_id,
                "town": town,
                "completion_reason": completion_reason,
                "progress_percent": max(row["route_progress_percent"] for row in rows),
                "final_distance_to_destination_m": rows[-1]["distance_to_destination_m"],
                "requested_start": {
                    "x": requested_start_transform.location.x,
                    "y": requested_start_transform.location.y,
                    "z": requested_start_transform.location.z,
                    "yaw": requested_start_transform.rotation.yaw,
                },
                "spawned_start": {
                    "x": start_transform.location.x,
                    "y": start_transform.location.y,
                    "z": start_transform.location.z,
                    "yaw": start_transform.rotation.yaw,
                },
            },
            "simulation": {
                "frames": len(rows),
                "duration_seconds": len(rows) * args.fixed_delta_seconds,
                "fixed_delta_seconds": args.fixed_delta_seconds,
                "inference_interval_frames": args.inference_interval,
                "traffic_vehicles_requested": args.traffic_vehicles,
                "seed": args.seed,
            },
            "events": events,
            "speed_mps": {
                "mean": float(np.mean([row["speed_mps"] for row in rows])),
                "p50": float(np.percentile([row["speed_mps"] for row in rows], 50)),
                "p99": float(np.percentile([row["speed_mps"] for row in rows], 99)),
            },
            "latency": {
                "steady_state_excluding_first_inference": {
                    key: summarize([
                        item["latency_ms"][key] for item in inference_rows[1:]
                    ])
                    for key in (
                        "infrastructure_forward", "ego_forward",
                        "combined_forward", "service_end_to_end",
                    )
                },
                "including_cold_start": {
                    key: summarize([
                        item["latency_ms"][key] for item in inference_rows
                    ])
                    for key in (
                        "infrastructure_forward", "ego_forward",
                        "combined_forward", "service_end_to_end",
                    )
                },
            },
            "rows": rows,
        }
        atomic_json(args.output, result)
        service_request(channel, {"op": "stop"})
    finally:
        try:
            channel.close()
            service_socket.close()
        except Exception:
            pass
        for sensor in sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
            except Exception:
                pass
        actor_ids = []
        for actor in reversed(sensors + actors):
            try:
                if actor.is_alive:
                    actor_ids.append(actor.id)
            except Exception:
                pass
        if actor_ids:
            try:
                client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in actor_ids])
            except Exception:
                pass
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
