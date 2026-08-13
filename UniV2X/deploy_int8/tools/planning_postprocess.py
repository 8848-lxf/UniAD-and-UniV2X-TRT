"""Deployment-side planning post-processing shared by CARLA services."""

import numpy as np


COLLISION_SCALE = 5.0 / 2.507


def _collision_objective(position, reference, occupied):
    delta_reference = position - reference
    value = float(np.dot(delta_reference, delta_reference))
    gradient = 2.0 * delta_reference
    if occupied.size:
        delta = position[None, :] - occupied
        penalty = COLLISION_SCALE * np.exp(
            -np.sum(delta * delta, axis=1) / 2.0
        )
        value += float(penalty.sum())
        gradient -= np.sum(penalty[:, None] * delta, axis=0)
    return value, gradient


def _optimize_planning_point(reference, occupied):
    if not occupied.size:
        return reference.copy()
    position = reference.astype(np.float64, copy=True)
    inverse_hessian = np.eye(2, dtype=np.float64)
    current_value, current_gradient = _collision_objective(
        position, reference, occupied
    )
    for _ in range(100):
        gradient_norm = float(np.linalg.norm(current_gradient))
        if gradient_norm < 1.0e-8:
            break
        direction = -inverse_hessian.dot(current_gradient)
        directional_derivative = float(direction.dot(current_gradient))
        if directional_derivative >= 0.0:
            direction = -current_gradient
            inverse_hessian = np.eye(2, dtype=np.float64)
            directional_derivative = -(gradient_norm * gradient_norm)

        step = 1.0
        candidate = position
        next_value = current_value
        next_gradient = current_gradient
        while step > 1.0e-10:
            candidate = position + step * direction
            next_value, next_gradient = _collision_objective(
                candidate, reference, occupied
            )
            if next_value <= (
                current_value + 1.0e-4 * step * directional_derivative
            ):
                break
            step *= 0.5
        if step <= 1.0e-10:
            break

        displacement = candidate - position
        gradient_delta = next_gradient - current_gradient
        curvature = float(displacement.dot(gradient_delta))
        if curvature > 1.0e-12:
            rho = 1.0 / curvature
            identity = np.eye(2, dtype=np.float64)
            left = identity - rho * np.outer(displacement, gradient_delta)
            right = identity - rho * np.outer(gradient_delta, displacement)
            inverse_hessian = (
                left.dot(inverse_hessian).dot(right)
                + rho * np.outer(displacement, displacement)
            )
        else:
            inverse_hessian = np.eye(2, dtype=np.float64)
        position = candidate
        current_value = next_value
        current_gradient = next_gradient
    return position


def _occupancy_array(occupancy):
    value = np.asarray(occupancy)
    if value.ndim == 5:
        value = value[0]
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError("unexpected occupancy shape: %r" % (value.shape,))
    return value


def optimize_collision(trajectory, occupancy, filter_range=5.0):
    """Match the C++ deployment BFGS collision optimizer."""
    reference = np.asarray(trajectory, dtype=np.float64)
    if reference.ndim == 3 and reference.shape[0] == 1:
        reference = reference[0]
    if reference.ndim != 2 or reference.shape[1] < 2:
        raise ValueError("unexpected trajectory shape: %r" % (reference.shape,))
    result = reference[:, :2].copy()
    occupancy = _occupancy_array(occupancy)
    horizon, height, width = occupancy.shape
    occupied_total = 0
    optimized_points = 0
    squared_range = filter_range * filter_range
    for timestep, point in enumerate(result):
        occupancy_timestep = min(timestep + 1, horizon - 1)
        pixels = np.argwhere(occupancy[occupancy_timestep] != 0)
        if not pixels.size:
            continue
        occupied = np.empty((pixels.shape[0], 2), dtype=np.float64)
        occupied[:, 0] = (pixels[:, 1] - height // 2) * 0.5 + 0.25
        occupied[:, 1] = (pixels[:, 0] - width // 2) * 0.5 + 0.25
        delta = point[None, :] - occupied
        occupied = occupied[np.sum(delta * delta, axis=1) < squared_range]
        occupied_total += int(occupied.shape[0])
        optimized = _optimize_planning_point(point, occupied)
        if not np.allclose(optimized, point, rtol=0.0, atol=1.0e-7):
            optimized_points += 1
        result[timestep] = optimized
    return result.astype(np.float32), {
        "collision_occupied_points": occupied_total,
        "collision_optimized_points": optimized_points,
    }


def optimize_drivable(trajectory, feasible_area):
    """Match UniV2X's deployed tail-freeze drivable-area correction."""
    original = np.asarray(trajectory, dtype=np.float32)
    if original.ndim == 3 and original.shape[0] == 1:
        original = original[0]
    area = np.asarray(feasible_area)
    while area.ndim > 2:
        area = area[0]
    if area.ndim != 2:
        raise ValueError("unexpected drivable shape: %r" % (area.shape,))

    height, width = area.shape
    bev = np.empty_like(original[:, :2], dtype=np.float32)
    bev[:, 0] = (-original[:, 1] + 51.2) / 102.4 * (height - 1)
    bev[:, 1] = (original[:, 0] + 51.2) / 102.4 * (width - 1)
    bev[:, 0] = np.clip(bev[:, 0], 0, height - 1)
    bev[:, 1] = np.clip(bev[:, 1], 0, width - 1)

    non_feasible = []
    for index, point in enumerate(bev):
        row, column = point.astype(np.int64)
        feasible_count = 0
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                candidate_row = row + row_delta
                candidate_column = column + column_delta
                if (
                    0 <= candidate_row < height
                    and 0 <= candidate_column < width
                ):
                    feasible_count += int(
                        area[candidate_row, candidate_column] == 1
                    )
                else:
                    feasible_count += 1
        if feasible_count < 2:
            non_feasible.append(index)

    if not non_feasible:
        return original[:, :2].copy(), {"drivable_adjusted_points": 0}
    consecutive = all(
        following == previous + 1
        for previous, following in zip(non_feasible[:-1], non_feasible[1:])
    )
    adjusted = bev.copy()
    adjusted_count = 0
    if consecutive and non_feasible[-1] == len(adjusted) - 1:
        for index in non_feasible:
            adjusted[index] = adjusted[index - 1]
            adjusted_count += 1
    result = np.empty_like(adjusted)
    result[:, 0] = adjusted[:, 1] * (102.4 / width) - 51.2
    result[:, 1] = -adjusted[:, 0] * (102.4 / height) + 51.2
    return result, {"drivable_adjusted_points": adjusted_count}


def postprocess_planning(trajectory, occupancy, drivable=None):
    raw = np.asarray(trajectory, dtype=np.float32)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    result, statistics = optimize_collision(raw, occupancy)
    if drivable is not None:
        result, drivable_statistics = optimize_drivable(result, drivable)
        statistics.update(drivable_statistics)
    delta = np.linalg.norm(result[:, :2] - raw[:, :2], axis=1)
    statistics.update({
        "raw_to_optimized_mean_l2_m": float(delta.mean()),
        "raw_to_optimized_max_l2_m": float(delta.max()),
    })
    return result, statistics
