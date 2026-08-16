#!/usr/bin/env python3
"""Capture a synchronized two-camera CARLA sequence for UniV2X replay."""

import argparse
import json
import math
import os
import queue
import random
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2010)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--capture-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


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
    deadline = time.time() + 10.0
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


def camera_blueprint(library):
    blueprint = library.find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", "1920")
    blueprint.set_attribute("image_size_y", "1080")
    blueprint.set_attribute("fov", "90")
    blueprint.set_attribute("sensor_tick", "0.0")
    return blueprint


def vehicle_ground_truth(world):
    records = []
    for actor in world.get_actors().filter("vehicle.*"):
        transform = actor.get_transform()
        extent = actor.bounding_box.extent
        records.append({
            "id": int(actor.id),
            "type_id": actor.type_id,
            "world_from_vehicle": transform_matrix(transform).tolist(),
            "bbox_extent_xyz": [extent.x, extent.y, extent.z],
        })
    return records


def spawn_traffic(world, traffic_manager, traffic_manager_port, spawn_points, count, seed):
    randomizer = random.Random(seed)
    candidates = list(spawn_points[1:])
    randomizer.shuffle(candidates)
    vehicles = []
    library = world.get_blueprint_library()
    blueprints = list(library.filter("vehicle.*"))
    for spawn_point in candidates:
        if len(vehicles) >= count:
            break
        blueprint = randomizer.choice(blueprints)
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", randomizer.choice(colors))
        actor = world.try_spawn_actor(blueprint, spawn_point)
        if actor is not None:
            actor.set_autopilot(True, traffic_manager_port)
            vehicles.append(actor)
    return vehicles


def main():
    args = parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise RuntimeError("output directory is not empty: %s" % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    original_settings = world.get_settings()
    traffic_manager_port = args.port + 6000
    traffic_manager = client.get_trafficmanager(traffic_manager_port)
    actors = []
    sensors = []
    event_counts = {"collision": 0, "lane_invasion": 0}
    records = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        spawn_points = world.get_map().get_spawn_points()
        if len(spawn_points) < 3:
            raise RuntimeError("CARLA map has fewer than three spawn points")
        randomizer = random.Random(args.seed)
        ego_blueprints = list(world.get_blueprint_library().filter("vehicle.tesla.model3"))
        if not ego_blueprints:
            ego_blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
        ego_spawn = randomizer.choice(spawn_points)
        ego = world.spawn_actor(randomizer.choice(ego_blueprints), ego_spawn)
        actors.append(ego)
        ego.set_autopilot(True, traffic_manager_port)

        traffic = spawn_traffic(
            world, traffic_manager, traffic_manager_port, spawn_points,
            args.vehicles, args.seed + 1
        )
        actors.extend(traffic)

        camera_bp = camera_blueprint(world.get_blueprint_library())
        ego_camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=1.8)),
            attach_to=ego,
        )
        sensors.append(ego_camera)

        waypoint = world.get_map().get_waypoint(ego_spawn.location)
        roadside_base = waypoint.next(25.0)
        roadside_waypoint = roadside_base[0] if roadside_base else waypoint
        roadside_transform = roadside_waypoint.transform
        yaw_radians = math.radians(roadside_transform.rotation.yaw)
        right_x = -math.sin(yaw_radians)
        right_y = math.cos(yaw_radians)
        infrastructure_transform = carla.Transform(
            carla.Location(
                x=roadside_transform.location.x + 5.0 * right_x,
                y=roadside_transform.location.y + 5.0 * right_y,
                z=roadside_transform.location.z + 5.5,
            ),
            carla.Rotation(
                pitch=-18.0,
                yaw=roadside_transform.rotation.yaw + 180.0,
                roll=0.0,
            ),
        )
        infrastructure_camera = world.spawn_actor(camera_bp, infrastructure_transform)
        sensors.append(infrastructure_camera)

        collision_bp = world.get_blueprint_library().find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego
        )
        collision_sensor.listen(
            lambda event: event_counts.__setitem__(
                "collision", event_counts["collision"] + 1
            )
        )
        sensors.append(collision_sensor)
        lane_bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=ego)
        lane_sensor.listen(
            lambda event: event_counts.__setitem__(
                "lane_invasion", event_counts["lane_invasion"] + 1
            )
        )
        sensors.append(lane_sensor)

        ego_queue = queue.Queue()
        infrastructure_queue = queue.Queue()
        ego_camera.listen(ego_queue.put)
        infrastructure_camera.listen(infrastructure_queue.put)

        for _ in range(20):
            world.tick()

        for sample_index in range(args.frames):
            requested_frame = None
            for _ in range(args.capture_interval):
                requested_frame = world.tick()
            snapshot = world.get_snapshot()
            if snapshot.frame != requested_frame:
                raise RuntimeError("world snapshot is not synchronized")
            ego_image = matching_measurement(
                ego_queue, requested_frame, "ego camera"
            )
            infrastructure_image = matching_measurement(
                infrastructure_queue, requested_frame, "infrastructure camera"
            )
            ego_transform = ego.get_transform()
            velocity = ego.get_velocity()
            acceleration = ego.get_acceleration()
            angular_velocity = ego.get_angular_velocity()
            control = ego.get_control()
            frame_name = "frame_%06d.npz" % sample_index
            np.savez_compressed(
                os.path.join(args.output_dir, frame_name),
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
            )
            records.append({
                "index": sample_index,
                "carla_frame": int(requested_frame),
                "timestamp_seconds": snapshot.timestamp.elapsed_seconds,
                "file": frame_name,
                "ego_control": {
                    "throttle": control.throttle,
                    "steer": control.steer,
                    "brake": control.brake,
                },
                "events_cumulative": dict(event_counts),
                "vehicles": vehicle_ground_truth(world),
            })
            print("captured %d/%d" % (sample_index + 1, args.frames), flush=True)

        manifest = {
            "schema": "univ2x_carla_two_camera_v1",
            "map": world.get_map().name,
            "ego_actor_id": int(ego.id),
            "frames": records,
            "capture": {
                "fixed_delta_seconds": args.fixed_delta_seconds,
                "capture_interval": args.capture_interval,
                "effective_interval_seconds": (
                    args.fixed_delta_seconds * args.capture_interval
                ),
                "camera": {"width": 1920, "height": 1080, "fov_degrees": 90.0},
                "traffic_vehicles_spawned": len(traffic),
                "seed": args.seed,
                "traffic_manager_port": traffic_manager_port,
            },
            "events": event_counts,
        }
        with open(os.path.join(args.output_dir, "manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2)
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception:
                pass
        # Attached sensors must be destroyed before their parent vehicle. CARLA
        # destroys them with the parent and aborts on a later duplicate destroy.
        for actor in reversed(sensors):
            try:
                actor.destroy()
            except Exception:
                pass
        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception:
                pass
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
