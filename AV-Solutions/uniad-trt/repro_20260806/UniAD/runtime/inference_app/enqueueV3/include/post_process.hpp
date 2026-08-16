/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <utility>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <sys/stat.h>
#include "uniad.hpp"

#ifdef UNIAD_USE_CASADI_COLLISION_OPTIMIZER
#include <casadi/casadi.hpp>
#endif

bool collision_optimization_enabled() {
    const char* value = std::getenv("UNIAD_COLLISION_OPTIMIZATION");
    return value != nullptr && std::string(value) != "0";
}

struct CollisionOptimizationAudit {
    std::uint64_t decode_calls = 0;
    std::uint64_t enabled_calls = 0;
    std::uint64_t frames_with_positive_occupancy = 0;
    std::uint64_t positive_occupancy_cells = 0;
    std::uint64_t frames_with_candidates = 0;
    std::uint64_t candidate_points = 0;
    std::uint64_t frames_modified = 0;
    std::uint64_t points_modified = 0;
    double max_point_delta_m = 0.0;
    std::uint64_t last_positive_occupancy_cells = 0;
    std::uint64_t last_candidate_points = 0;
    std::uint64_t last_points_modified = 0;
    double last_max_point_delta_m = 0.0;
};

CollisionOptimizationAudit& collision_optimization_audit() {
    static CollisionOptimizationAudit audit;
    return audit;
}

struct CollisionObjective {
    double value;
    std::array<double, 2> gradient;
};

CollisionObjective collision_objective(
    const std::array<double, 2>& position,
    const std::array<double, 2>& reference,
    const std::vector<std::array<double, 2>>& occupied) {
    constexpr double collision_scale = 5.0 / 2.507;
    const double reference_x = position[0] - reference[0];
    const double reference_y = position[1] - reference[1];
    CollisionObjective result{
        reference_x * reference_x + reference_y * reference_y,
        {2.0 * reference_x, 2.0 * reference_y}};
    for (const auto& point : occupied) {
        const double dx = position[0] - point[0];
        const double dy = position[1] - point[1];
        const double penalty = collision_scale * std::exp(-(dx * dx + dy * dy) / 2.0);
        result.value += penalty;
        result.gradient[0] -= penalty * dx;
        result.gradient[1] -= penalty * dy;
    }
    return result;
}

std::array<double, 2> optimize_planning_point(
    const std::array<double, 2>& reference,
    const std::vector<std::array<double, 2>>& occupied) {
    if (occupied.empty()) return reference;
    std::array<double, 2> position = reference;
    std::array<double, 4> inverse_hessian = {1.0, 0.0, 0.0, 1.0};
    CollisionObjective current = collision_objective(position, reference, occupied);
    for (int iteration = 0; iteration < 100; ++iteration) {
        const double gradient_norm = std::hypot(
            current.gradient[0], current.gradient[1]);
        if (gradient_norm < 1.0e-8) break;
        std::array<double, 2> direction = {
            -(inverse_hessian[0] * current.gradient[0]
              + inverse_hessian[1] * current.gradient[1]),
            -(inverse_hessian[2] * current.gradient[0]
              + inverse_hessian[3] * current.gradient[1])};
        double directional_derivative =
            direction[0] * current.gradient[0]
            + direction[1] * current.gradient[1];
        if (directional_derivative >= 0.0) {
            direction = {-current.gradient[0], -current.gradient[1]};
            inverse_hessian = {1.0, 0.0, 0.0, 1.0};
            directional_derivative = -gradient_norm * gradient_norm;
        }

        double step = 1.0;
        std::array<double, 2> candidate{};
        CollisionObjective next{};
        while (step > 1.0e-10) {
            candidate = {
                position[0] + step * direction[0],
                position[1] + step * direction[1]};
            next = collision_objective(candidate, reference, occupied);
            if (next.value <= current.value
                    + 1.0e-4 * step * directional_derivative) break;
            step *= 0.5;
        }
        if (step <= 1.0e-10) break;

        const std::array<double, 2> displacement = {
            candidate[0] - position[0], candidate[1] - position[1]};
        const std::array<double, 2> gradient_delta = {
            next.gradient[0] - current.gradient[0],
            next.gradient[1] - current.gradient[1]};
        const double curvature =
            displacement[0] * gradient_delta[0]
            + displacement[1] * gradient_delta[1];
        if (curvature > 1.0e-12) {
            const double rho = 1.0 / curvature;
            const std::array<double, 4> left = {
                1.0 - rho * displacement[0] * gradient_delta[0],
                -rho * displacement[0] * gradient_delta[1],
                -rho * displacement[1] * gradient_delta[0],
                1.0 - rho * displacement[1] * gradient_delta[1]};
            const std::array<double, 4> right = {
                1.0 - rho * gradient_delta[0] * displacement[0],
                -rho * gradient_delta[0] * displacement[1],
                -rho * gradient_delta[1] * displacement[0],
                1.0 - rho * gradient_delta[1] * displacement[1]};
            std::array<double, 4> intermediate = {
                left[0] * inverse_hessian[0] + left[1] * inverse_hessian[2],
                left[0] * inverse_hessian[1] + left[1] * inverse_hessian[3],
                left[2] * inverse_hessian[0] + left[3] * inverse_hessian[2],
                left[2] * inverse_hessian[1] + left[3] * inverse_hessian[3]};
            inverse_hessian = {
                intermediate[0] * right[0] + intermediate[1] * right[2]
                    + rho * displacement[0] * displacement[0],
                intermediate[0] * right[1] + intermediate[1] * right[3]
                    + rho * displacement[0] * displacement[1],
                intermediate[2] * right[0] + intermediate[3] * right[2]
                    + rho * displacement[1] * displacement[0],
                intermediate[2] * right[1] + intermediate[3] * right[3]
                    + rho * displacement[1] * displacement[1]};
        } else {
            inverse_hessian = {1.0, 0.0, 0.0, 1.0};
        }
        position = candidate;
        current = next;
    }
    return position;
}

#ifdef UNIAD_USE_CASADI_COLLISION_OPTIMIZER
std::vector<std::pair<float, float>> optimize_planning_trajectory_casadi(
    const std::vector<std::pair<float, float>>& reference,
    const std::vector<std::vector<std::array<double, 2>>>& occupied_by_timestep) {
    const casadi_int trajectory_len = static_cast<casadi_int>(reference.size());
    casadi::Opti optimizer;
    casadi::MX state = optimizer.variable(2, trajectory_len);
    casadi::MX reference_parameter = optimizer.parameter(2, trajectory_len);
    casadi::MX cost = casadi::MX::sumsqr(reference_parameter - state);
    constexpr double collision_scale = 5.0 / 2.507;
    for (casadi_int timestep = 0; timestep < trajectory_len; ++timestep) {
        const casadi::MX x = state(0, timestep);
        const casadi::MX y = state(1, timestep);
        for (const auto& point : occupied_by_timestep.at(timestep)) {
            const casadi::MX dx = x - point[0];
            const casadi::MX dy = y - point[1];
            cost += collision_scale * casadi::MX::exp(-(dx * dx + dy * dy) / 2.0);
        }
    }
    optimizer.minimize(cost);
    casadi::Dict options;
    options["ipopt.print_level"] = 0;
    options["print_time"] = 0;
    options["ipopt.sb"] = std::string("yes");
    optimizer.solver("ipopt", options);

    casadi::DM reference_value = casadi::DM::zeros(2, trajectory_len);
    for (casadi_int timestep = 0; timestep < trajectory_len; ++timestep) {
        reference_value(0, timestep) = reference.at(timestep).first;
        reference_value(1, timestep) = reference.at(timestep).second;
    }
    optimizer.set_value(reference_parameter, reference_value);
    optimizer.set_initial(state, reference_value);
    const casadi::OptiSol solution = optimizer.solve();
    const casadi::DM optimized = solution.value(state);

    std::vector<std::pair<float, float>> result;
    result.reserve(reference.size());
    for (casadi_int timestep = 0; timestep < trajectory_len; ++timestep) {
        result.emplace_back(
            static_cast<float>(static_cast<double>(optimized(0, timestep))),
            static_cast<float>(static_cast<double>(optimized(1, timestep))));
    }
    return result;
}
#endif

std::vector<std::pair<float, float>> decode_planning_traj(const UniAD::KernelOutput& output_instance) {
    auto& audit = collision_optimization_audit();
    ++audit.decode_calls;
    audit.last_positive_occupancy_cells = 0;
    audit.last_candidate_points = 0;
    audit.last_points_modified = 0;
    audit.last_max_point_delta_m = 0.0;
    std::vector<std::pair<float, float>> planning_traj;
    for (size_t i=0; i<output_instance.outs_planning.size(); i+=2) {
        planning_traj.push_back({output_instance.outs_planning[i], output_instance.outs_planning[i+1]});
    }
    if (!collision_optimization_enabled()) return planning_traj;
    ++audit.enabled_calls;
    const std::vector<std::pair<float, float>> raw_planning_traj = planning_traj;

    const auto shape = output_instance.output_shapes.at("seg_out");
    if (shape.size() != 5 || shape[0] != 1 || shape[2] != 1) {
        throw std::runtime_error("Unexpected seg_out shape for collision optimization");
    }
    const int horizon = static_cast<int>(shape[1]);
    const int height = static_cast<int>(shape[3]);
    const int width = static_cast<int>(shape[4]);
    std::size_t positive_cells = 0;
    for (const int32_t value : output_instance.seg_out) {
        if (value != 0) ++positive_cells;
    }
    audit.positive_occupancy_cells += positive_cells;
    audit.last_positive_occupancy_cells = positive_cells;
    if (positive_cells > 0) ++audit.frames_with_positive_occupancy;
    std::vector<std::vector<std::array<double, 2>>> occupied_by_timestep(
        planning_traj.size());
    size_t total_occupied = 0;
    for (size_t timestep = 0; timestep < planning_traj.size(); ++timestep) {
        const int occupancy_timestep = std::min(
            static_cast<int>(timestep) + 1, horizon - 1);
        const size_t offset = static_cast<size_t>(occupancy_timestep)
            * height * width;
        const std::array<double, 2> reference = {
            planning_traj[timestep].first, planning_traj[timestep].second};
        auto& occupied = occupied_by_timestep[timestep];
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                if (output_instance.seg_out[offset + row * width + column] == 0) continue;
                const std::array<double, 2> point = {
                    (column - height / 2) * 0.5 + 0.25,
                    (row - width / 2) * 0.5 + 0.25};
                const double dx = reference[0] - point[0];
                const double dy = reference[1] - point[1];
                if (dx * dx + dy * dy < 25.0) occupied.push_back(point);
            }
        }
        total_occupied += occupied.size();
#ifndef UNIAD_USE_CASADI_COLLISION_OPTIMIZER
        const auto optimized = optimize_planning_point(reference, occupied);
        planning_traj[timestep] = {
            static_cast<float>(optimized[0]), static_cast<float>(optimized[1])};
#endif
    }
    audit.candidate_points += total_occupied;
    audit.last_candidate_points = total_occupied;
    if (total_occupied > 0) ++audit.frames_with_candidates;
#ifdef UNIAD_USE_CASADI_COLLISION_OPTIMIZER
    if (total_occupied > 0) {
        planning_traj = optimize_planning_trajectory_casadi(
            planning_traj, occupied_by_timestep);
    }
#endif
    bool frame_modified = false;
    std::size_t frame_points_modified = 0;
    double frame_max_point_delta_m = 0.0;
    for (std::size_t index = 0; index < planning_traj.size(); ++index) {
        const double dx = static_cast<double>(planning_traj[index].first)
            - raw_planning_traj[index].first;
        const double dy = static_cast<double>(planning_traj[index].second)
            - raw_planning_traj[index].second;
        const double delta = std::hypot(dx, dy);
        audit.max_point_delta_m = std::max(audit.max_point_delta_m, delta);
        frame_max_point_delta_m = std::max(frame_max_point_delta_m, delta);
        if (delta > 1.0e-7) {
            frame_modified = true;
            ++audit.points_modified;
            ++frame_points_modified;
        }
    }
    audit.last_points_modified = frame_points_modified;
    audit.last_max_point_delta_m = frame_max_point_delta_m;
    if (frame_modified) ++audit.frames_modified;
    return planning_traj;
}

std::vector<std::pair<float, float>> decode_raw_planning_traj(
    const UniAD::KernelOutput& output_instance) {
    std::vector<std::pair<float, float>> planning_traj;
    for (size_t i = 0; i < output_instance.outs_planning.size(); i += 2) {
        planning_traj.push_back({
            output_instance.outs_planning[i],
            output_instance.outs_planning[i + 1]});
    }
    return planning_traj;
}

std::string decode_command(const UniAD::KernelInput& input_instance) {
    std::unordered_map<int, std::string> command_map = {
        {0, "TURN RIGHT"},
        {1, "TURN LEFT"},
        {2, "KEEP FORWARD"}
    };
    int command_idx = (int)(input_instance.command[0]);
    return command_map[command_idx];
}

std::vector<std::vector<float>> decode_bbox(const UniAD::KernelOutput& output_instance) {
    std::vector<std::vector<float>> pred_bbox;
    int total_number_bbox = output_instance.output_shapes.at("scores")[0];
    for (size_t i=0; i<total_number_bbox; ++i) {
        if (output_instance.scores[i] < 0.25) continue;
        pred_bbox.push_back({
            output_instance.bboxes_dict_bboxes[9*i+0], // x
            output_instance.bboxes_dict_bboxes[9*i+1], // y
            output_instance.bboxes_dict_bboxes[9*i+2] + output_instance.bboxes_dict_bboxes[9*i+5]/2., // z
            output_instance.bboxes_dict_bboxes[9*i+3], // w
            output_instance.bboxes_dict_bboxes[9*i+4], // l
            output_instance.bboxes_dict_bboxes[9*i+5], // h
            output_instance.bboxes_dict_bboxes[9*i+6], // yaw
            output_instance.bboxes_dict_bboxes[9*i+7], // vx
            output_instance.bboxes_dict_bboxes[9*i+8], // vy
            output_instance.labels[i], // label
            output_instance.scores[i], // scores
        });
    }
    return pred_bbox;
}
