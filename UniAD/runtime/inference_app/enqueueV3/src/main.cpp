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
#include <dlfcn.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>
#include <fstream>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <sys/stat.h>
#include "uniad.hpp"
#include "visualize.hpp"
#include "tensor.hpp"
#include "pre_process.hpp"
#include "post_process.hpp"

void readBinFileToVec(const std::string& filename, void* vec, std::vector<TRT_INT_TYPE>& vec_shape, const std::vector<TRT_INT_TYPE>& ref_shape, const std::size_t vec_dsize) {
    std::ifstream bin_file(filename, std::ios::binary);
    if (!bin_file) {
        printf("[ERROR] Could not read file at: %s\n", filename.c_str());
        return;
    }
    bin_file.seekg(0, bin_file.end);
    int file_length = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    int dynamic_idx = -1;
    for (size_t shape_id=0; shape_id<vec_shape.size(); ++shape_id) {
        if (vec_shape[shape_id] < 0) dynamic_idx = shape_id;
    }
    if (dynamic_idx < 0) {
        if (file_length != vec_dsize*std::accumulate(vec_shape.begin(), vec_shape.end(), 1, std::multiplies<int>())) {
            printf("[ERROR] File size not match at: %s\n", filename.c_str());
            return;
        }
    } else {
        // dynamic shape
        vec_shape[dynamic_idx] = 1;
        int actual_shape = file_length / (vec_dsize*std::accumulate(vec_shape.begin(), vec_shape.end(), 1, std::multiplies<int>()));
        int error_ = file_length % (vec_dsize*std::accumulate(vec_shape.begin(), vec_shape.end(), 1, std::multiplies<int>()));
        if (error_ > 0) {
            printf("[ERROR] Invalid dynamic shape %d (residual %d) at position %d for %s\n", actual_shape, error_, dynamic_idx, filename.c_str());
            return;
        }
        if (actual_shape > ref_shape[dynamic_idx]) {
            printf("[ERROR] Invalid dynamic shape %d (maximum %d) at position %d for %s\n", actual_shape, ref_shape[dynamic_idx], dynamic_idx, filename.c_str());
            return;
        }
        vec_shape[dynamic_idx] = actual_shape;
    }
    bin_file.read((char *)vec, file_length);
    bin_file.close();
    return;
}

std::vector<std::vector<std::string>> parse_input_info(const std::string& input_pth, size_t num_inputs, int bias) {
    std::string file_name = input_pth;
    if (file_name[file_name.size()-1]!='/') file_name += "/";
    file_name += "info.txt";
    std::ifstream info_file(file_name);
    std::string info_str;
    std::vector<std::vector<std::string>> infos_vec;
    for (int _i=0; _i<bias; ++_i) std::getline(info_file, info_str);
    while (std::getline(info_file, info_str) && infos_vec.size()<num_inputs) {
        std::vector<std::string> info_;
        while (info_str.find(';') != std::string::npos) {
            size_t length = info_str.find(';');
            info_.push_back(info_str.substr(0, length));
            info_str = info_str.substr(length+1);
        }
        infos_vec.push_back(info_);
    }
    info_file.close();
    return infos_vec;
}

void load_input(const std::string& input_pth, std::vector<UniAD::KernelInput>& inputs, int num_inputs) {
    printf("[INFO] Loading input from %s\n", input_pth.c_str());
    std::vector<std::string> input_to_load{
        "timestamp", "l2g_r_mat", "l2g_t", "command",
        "img_metas_can_bus", "img_metas_lidar2img"
    };
    std::vector<float> curr_scene = std::vector<float>(32, 0);
    std::vector<TRT_INT_TYPE> scene_shape = std::vector<TRT_INT_TYPE>(1, 32);
    const UniAD::KernelParams kernel_params_ref;
    for (int file_id=0; file_id<num_inputs; ++file_id) {
        UniAD::KernelInput input_instance;
        std::string file_name = input_pth;
        if (file_name[file_name.size()-1] != '/') file_name += "/";
        for (std::string input_type : input_to_load) {
            if (input_instance.input_shapes.find(input_type) == input_instance.input_shapes.cend()) {
                // static shape
                input_instance.input_shapes[input_type] = kernel_params_ref.input_max_shapes.at(input_type);
            }
            readBinFileToVec(
                file_name+input_type+"/"+std::to_string(file_id)+".bin", 
                input_instance.data_ptrs[input_type],
                input_instance.input_shapes[input_type],
                kernel_params_ref.input_max_shapes.at(input_type),
                kernel_params_ref.input_sizes.at(input_type)
            );
        }
        std::vector<float> input_scene = std::vector<float>(32, 0);
        readBinFileToVec(
            file_name+"img_metas_scene_token/"+std::to_string(file_id)+".bin", 
            input_scene.data(),
            scene_shape,
            scene_shape,
            sizeof(float)
        );
        if (input_scene != curr_scene) {
            input_instance.use_prev_bev[0] = 0.;
            printf("[INFO] Scene changed at file id %d.\n", file_id);
            curr_scene = input_scene;
        } else input_instance.use_prev_bev[0] = 1.;
        input_instance.input_shapes["use_prev_bev"] = kernel_params_ref.input_max_shapes.at("use_prev_bev");
        inputs.push_back(input_instance);
    }
    return;
}

static void visualize(const std::vector<unsigned char*> images, const UniAD::KernelInput& input_instance, const UniAD::KernelOutput& output_instance,
                      const std::vector<std::pair<float, float>>& planning_traj, const std::string& save_path,
                      cudaStream_t stream) {
    std::string command = decode_command(input_instance);
    std::vector<std::vector<float>> pred_bbox = decode_bbox(output_instance);

    int lidar_size = 1200;
    int content_width = lidar_size + 900;
    nv::ImageArtistParameter image_artist_param;
    image_artist_param.num_camera = images.size();
    image_artist_param.image_width = 1600;
    image_artist_param.image_height = 900;
    image_artist_param.image_stride = image_artist_param.image_width * 3;
    for (size_t i=0; i<input_instance.img_metas_lidar2img.size(); i+=4) {
        nvtype::Float4 transform_vec(input_instance.img_metas_lidar2img[i], input_instance.img_metas_lidar2img[i+1],
                                    input_instance.img_metas_lidar2img[i+2], input_instance.img_metas_lidar2img[i+3]);
        image_artist_param.viewport_nx4x4.push_back(transform_vec);
    }
    int gap = 0;
    int camera_width = content_width/3;
    int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
    int content_height = 2*camera_height + 3*content_width/4;
    nv::SceneArtistParameter scene_artist_param;
    scene_artist_param.width = content_width;
    scene_artist_param.height = content_height;
    scene_artist_param.stride = scene_artist_param.width * 3;

    nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
    scene_device_image.memset(0x00, stream);

    scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
    auto scene = nv::create_scene_artist(scene_artist_param);

    nv::BEVArtistParameter bev_artist_param;
    bev_artist_param.image_width = content_width;
    bev_artist_param.image_height = content_height;
    bev_artist_param.rotate_x = 70.0f;
    bev_artist_param.norm_size = lidar_size * 0.5f;
    bev_artist_param.cx = content_width * 0.5f;
    bev_artist_param.cy = content_height * 0.5f + camera_height;
    bev_artist_param.image_stride = scene_artist_param.stride;

    auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
    bev_visualizer->draw_ego();
    for (int r=15; r<=60; r+=15) bev_visualizer->draw_circle(0, 0, r);
    bev_visualizer->draw_planning_traj(planning_traj, command);
    bev_visualizer->draw_prediction(pred_bbox, true);
    bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

    int offset_cameras[][3] = {
        {camera_width, 0, 0},
        {camera_width*2, 0, 0},
        {0, 0, 0},
        {camera_width, camera_height, 1},
        {0, camera_height, 1},
        {camera_width*2, camera_height, 1}};

    auto visualizer = nv::create_image_artist(image_artist_param);
    for (size_t icamera = 0; icamera < images.size(); ++icamera) {
        int ox = offset_cameras[icamera][0];
        int oy = offset_cameras[icamera][1];
        bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
        visualizer->draw_prediction(icamera, pred_bbox, xflip);
        visualizer->draw_planning_traj(icamera, planning_traj, xflip);

        nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
        device_image.copy_from_host(images[icamera], stream);

        if (xflip) {
            auto clone = device_image.clone(stream);
            scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(),
                        device_image.size(1) * 3, stream);
            checkRuntime(cudaStreamSynchronize(stream));
        }
        visualizer->apply(device_image.ptr<unsigned char>(), stream);

        scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1),
                        device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
        checkRuntime(cudaStreamSynchronize(stream));
    }

    stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3,
                    scene_device_image.to_host(stream).ptr(), 100);
}

std::shared_ptr<UniAD::Kernel> create_kernel(const std::string& engine_pth) {
    UniAD::KernelParams params;
    params.trt_engine = engine_pth;
    std::shared_ptr<UniAD::Kernel> instance = std::make_shared<UniAD::KernelImplement>();
    if (instance->init(params) != 0) return nullptr;
    return instance;
}

template <typename T>
void assign_temporal_track_tensor(
    std::vector<T>& destination,
    std::vector<TRT_INT_TYPE>& input_shape,
    const std::vector<T>& source,
    const std::vector<TRT_INT_TYPE>& output_shape,
    int fixed_track_count,
    const char* tensor_name) {
    if (output_shape.empty() || output_shape[0] < 0) {
        fprintf(stderr, "[ERROR] Invalid temporal output shape for %s.\n", tensor_name);
        std::abort();
    }
    const size_t actual_rows = static_cast<size_t>(output_shape[0]);
    const size_t row_width = std::accumulate(
        output_shape.begin() + 1, output_shape.end(), size_t{1}, std::multiplies<size_t>());
    const size_t actual_elements = actual_rows * row_width;
    if (actual_elements > source.size()) {
        fprintf(stderr, "[ERROR] Temporal output %s exceeds its host buffer.\n", tensor_name);
        std::abort();
    }
    if (fixed_track_count > 0 && actual_rows > static_cast<size_t>(fixed_track_count)) {
        fprintf(stderr, "[ERROR] Temporal output %s has %zu rows, exceeding fixed capacity %d.\n",
                tensor_name, actual_rows, fixed_track_count);
        std::abort();
    }
    destination.assign(source.begin(), source.begin() + actual_elements);
    input_shape = output_shape;
    if (fixed_track_count > 0) {
        destination.resize(static_cast<size_t>(fixed_track_count) * row_width, static_cast<T>(-10000));
        input_shape[0] = fixed_track_count;
    }
}

template <typename T>
void pad_initial_track_tensor(
    std::vector<T>& values,
    std::vector<TRT_INT_TYPE>& shape,
    int fixed_track_count,
    const char* tensor_name) {
    const size_t actual_rows = static_cast<size_t>(shape.at(0));
    const size_t row_width = std::accumulate(
        shape.begin() + 1, shape.end(), size_t{1}, std::multiplies<size_t>());
    if (actual_rows > static_cast<size_t>(fixed_track_count)) {
        fprintf(stderr, "[ERROR] Initial temporal input %s has %zu rows, exceeding fixed capacity %d.\n",
                tensor_name, actual_rows, fixed_track_count);
        std::abort();
    }
    std::fill(
        values.begin() + actual_rows * row_width,
        values.begin() + static_cast<size_t>(fixed_track_count) * row_width,
        static_cast<T>(-10000));
    shape[0] = fixed_track_count;
}

void set_fixed_track_input_shapes(UniAD::KernelInput& input, int fixed_track_count) {
    if (fixed_track_count <= 0) return;
    pad_initial_track_tensor(input.prev_track_intances0, input.input_shapes.at("prev_track_intances0"), fixed_track_count, "prev_track_intances0");
    pad_initial_track_tensor(input.prev_track_intances1, input.input_shapes.at("prev_track_intances1"), fixed_track_count, "prev_track_intances1");
    pad_initial_track_tensor(input.prev_track_intances3, input.input_shapes.at("prev_track_intances3"), fixed_track_count, "prev_track_intances3");
    pad_initial_track_tensor(input.prev_track_intances4, input.input_shapes.at("prev_track_intances4"), fixed_track_count, "prev_track_intances4");
    pad_initial_track_tensor(input.prev_track_intances5, input.input_shapes.at("prev_track_intances5"), fixed_track_count, "prev_track_intances5");
    pad_initial_track_tensor(input.prev_track_intances6, input.input_shapes.at("prev_track_intances6"), fixed_track_count, "prev_track_intances6");
    pad_initial_track_tensor(input.prev_track_intances8, input.input_shapes.at("prev_track_intances8"), fixed_track_count, "prev_track_intances8");
    pad_initial_track_tensor(input.prev_track_intances9, input.input_shapes.at("prev_track_intances9"), fixed_track_count, "prev_track_intances9");
    pad_initial_track_tensor(input.prev_track_intances11, input.input_shapes.at("prev_track_intances11"), fixed_track_count, "prev_track_intances11");
    pad_initial_track_tensor(input.prev_track_intances12, input.input_shapes.at("prev_track_intances12"), fixed_track_count, "prev_track_intances12");
    pad_initial_track_tensor(input.prev_track_intances13, input.input_shapes.at("prev_track_intances13"), fixed_track_count, "prev_track_intances13");
}

void temporal_info_assign(
    UniAD::KernelInput& input_t,
    const UniAD::KernelOutput& output_t_1,
    int fixed_track_count) {
    assign_temporal_track_tensor(
        input_t.prev_track_intances0, input_t.input_shapes["prev_track_intances0"],
        output_t_1.prev_track_intances0_out, output_t_1.output_shapes.at("prev_track_intances0_out"),
        fixed_track_count, "prev_track_intances0");
    input_t.data_ptrs["prev_track_intances0"] = input_t.prev_track_intances0.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances1, input_t.input_shapes["prev_track_intances1"],
        output_t_1.prev_track_intances1_out, output_t_1.output_shapes.at("prev_track_intances1_out"),
        fixed_track_count, "prev_track_intances1");
    input_t.data_ptrs["prev_track_intances1"] = input_t.prev_track_intances1.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances3, input_t.input_shapes["prev_track_intances3"],
        output_t_1.prev_track_intances3_out, output_t_1.output_shapes.at("prev_track_intances3_out"),
        fixed_track_count, "prev_track_intances3");
    input_t.data_ptrs["prev_track_intances3"] = input_t.prev_track_intances3.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances4, input_t.input_shapes["prev_track_intances4"],
        output_t_1.prev_track_intances4_out, output_t_1.output_shapes.at("prev_track_intances4_out"),
        fixed_track_count, "prev_track_intances4");
    input_t.data_ptrs["prev_track_intances4"] = input_t.prev_track_intances4.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances5, input_t.input_shapes["prev_track_intances5"],
        output_t_1.prev_track_intances5_out, output_t_1.output_shapes.at("prev_track_intances5_out"),
        fixed_track_count, "prev_track_intances5");
    input_t.data_ptrs["prev_track_intances5"] = input_t.prev_track_intances5.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances6, input_t.input_shapes["prev_track_intances6"],
        output_t_1.prev_track_intances6_out, output_t_1.output_shapes.at("prev_track_intances6_out"),
        fixed_track_count, "prev_track_intances6");
    input_t.data_ptrs["prev_track_intances6"] = input_t.prev_track_intances6.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances8, input_t.input_shapes["prev_track_intances8"],
        output_t_1.prev_track_intances8_out, output_t_1.output_shapes.at("prev_track_intances8_out"),
        fixed_track_count, "prev_track_intances8");
    input_t.data_ptrs["prev_track_intances8"] = input_t.prev_track_intances8.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances9, input_t.input_shapes["prev_track_intances9"],
        output_t_1.prev_track_intances9_out, output_t_1.output_shapes.at("prev_track_intances9_out"),
        fixed_track_count, "prev_track_intances9");
    input_t.data_ptrs["prev_track_intances9"] = input_t.prev_track_intances9.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances11, input_t.input_shapes["prev_track_intances11"],
        output_t_1.prev_track_intances11_out, output_t_1.output_shapes.at("prev_track_intances11_out"),
        fixed_track_count, "prev_track_intances11");
    input_t.data_ptrs["prev_track_intances11"] = input_t.prev_track_intances11.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances12, input_t.input_shapes["prev_track_intances12"],
        output_t_1.prev_track_intances12_out, output_t_1.output_shapes.at("prev_track_intances12_out"),
        fixed_track_count, "prev_track_intances12");
    input_t.data_ptrs["prev_track_intances12"] = input_t.prev_track_intances12.data();

    assign_temporal_track_tensor(
        input_t.prev_track_intances13, input_t.input_shapes["prev_track_intances13"],
        output_t_1.prev_track_intances13_out, output_t_1.output_shapes.at("prev_track_intances13_out"),
        fixed_track_count, "prev_track_intances13");
    input_t.data_ptrs["prev_track_intances13"] = input_t.prev_track_intances13.data();

    input_t.prev_timestamp.assign(output_t_1.prev_timestamp_out.begin(), output_t_1.prev_timestamp_out.end());
    input_t.data_ptrs["prev_timestamp"] = input_t.prev_timestamp.data();
    input_t.input_shapes["prev_timestamp"] = output_t_1.output_shapes.at("prev_timestamp_out");

    input_t.prev_l2g_r_mat.assign(output_t_1.prev_l2g_r_mat_out.begin(), output_t_1.prev_l2g_r_mat_out.end());
    input_t.data_ptrs["prev_l2g_r_mat"] = input_t.prev_l2g_r_mat.data();
    input_t.input_shapes["prev_l2g_r_mat"] = output_t_1.output_shapes.at("prev_l2g_r_mat_out");

    input_t.prev_l2g_t.assign(output_t_1.prev_l2g_t_out.begin(), output_t_1.prev_l2g_t_out.end());
    input_t.data_ptrs["prev_l2g_t"] = input_t.prev_l2g_t.data();
    input_t.input_shapes["prev_l2g_t"] = output_t_1.output_shapes.at("prev_l2g_t_out");

    input_t.prev_bev.assign(output_t_1.bev_embed.begin(), output_t_1.bev_embed.end());
    input_t.data_ptrs["prev_bev"] = input_t.prev_bev.data();
    input_t.input_shapes["prev_bev"] = output_t_1.output_shapes.at("bev_embed");

    input_t.max_obj_id.assign(output_t_1.max_obj_id_out.begin(), output_t_1.max_obj_id_out.end());
    input_t.data_ptrs["max_obj_id"] = input_t.max_obj_id.data();
    input_t.input_shapes["max_obj_id"] = output_t_1.output_shapes.at("max_obj_id_out");
    return;
}

static void relink_input(UniAD::KernelInput& input) {
    input.data_ptrs["prev_track_intances0"] = input.prev_track_intances0.data();
    input.data_ptrs["prev_track_intances1"] = input.prev_track_intances1.data();
    input.data_ptrs["prev_track_intances3"] = input.prev_track_intances3.data();
    input.data_ptrs["prev_track_intances4"] = input.prev_track_intances4.data();
    input.data_ptrs["prev_track_intances5"] = input.prev_track_intances5.data();
    input.data_ptrs["prev_track_intances6"] = input.prev_track_intances6.data();
    input.data_ptrs["prev_track_intances8"] = input.prev_track_intances8.data();
    input.data_ptrs["prev_track_intances9"] = input.prev_track_intances9.data();
    input.data_ptrs["prev_track_intances11"] = input.prev_track_intances11.data();
    input.data_ptrs["prev_track_intances12"] = input.prev_track_intances12.data();
    input.data_ptrs["prev_track_intances13"] = input.prev_track_intances13.data();
    input.data_ptrs["prev_timestamp"] = input.prev_timestamp.data();
    input.data_ptrs["prev_l2g_r_mat"] = input.prev_l2g_r_mat.data();
    input.data_ptrs["prev_l2g_t"] = input.prev_l2g_t.data();
    input.data_ptrs["prev_bev"] = input.prev_bev.data();
    input.data_ptrs["timestamp"] = input.timestamp.data();
    input.data_ptrs["l2g_r_mat"] = input.l2g_r_mat.data();
    input.data_ptrs["l2g_t"] = input.l2g_t.data();
    input.data_ptrs["img"] = input.img.data();
    input.data_ptrs["img_metas_can_bus"] = input.img_metas_can_bus.data();
    input.data_ptrs["img_metas_lidar2img"] = input.img_metas_lidar2img.data();
    input.data_ptrs["command"] = input.command.data();
    input.data_ptrs["use_prev_bev"] = input.use_prev_bev.data();
    input.data_ptrs["max_obj_id"] = input.max_obj_id.data();
}

static bool load_input_frame(
    const std::string& input_pth,
    int frame_id,
    UniAD::KernelInput& input,
    std::vector<float>& current_scene,
    bool& have_scene) {
    const std::vector<std::string> input_to_load{
        "timestamp", "l2g_r_mat", "l2g_t", "command",
        "img_metas_can_bus", "img_metas_lidar2img"};
    const UniAD::KernelParams params;
    std::string root = input_pth;
    if (root.back() != '/') root += "/";

    for (const std::string& input_type : input_to_load) {
        input.input_shapes[input_type] = params.input_max_shapes.at(input_type);
        readBinFileToVec(
            root + input_type + "/" + std::to_string(frame_id) + ".bin",
            input.data_ptrs[input_type],
            input.input_shapes[input_type],
            params.input_max_shapes.at(input_type),
            params.input_sizes.at(input_type));
    }

    std::vector<float> input_scene(32, 0.0f);
    std::vector<TRT_INT_TYPE> scene_shape{32};
    readBinFileToVec(
        root + "img_metas_scene_token/" + std::to_string(frame_id) + ".bin",
        input_scene.data(), scene_shape, scene_shape, sizeof(float));
    const bool scene_changed = !have_scene || input_scene != current_scene;
    input.use_prev_bev[0] = scene_changed ? 0 : 1;
    input.input_shapes["use_prev_bev"] = params.input_max_shapes.at("use_prev_bev");
    if (scene_changed) {
        current_scene = input_scene;
        have_scene = true;
    }
    relink_input(input);
    return scene_changed;
}

static bool valid_images(const std::vector<unsigned char*>& images) {
    if (images.size() != 6) return false;
    for (const unsigned char* image : images) {
        if (image == nullptr) return false;
    }
    return true;
}

static bool finite_planning(
    const std::vector<std::pair<float, float>>& planning) {
    return std::all_of(
        planning.begin(), planning.end(),
        [](const std::pair<float, float>& point) {
            return std::isfinite(point.first) && std::isfinite(point.second);
        });
}

struct LatencySummary {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

static double percentile(const std::vector<double>& sorted, double quantile) {
    if (sorted.empty()) return 0.0;
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

static LatencySummary summarize(const std::vector<double>& samples) {
    LatencySummary result;
    if (samples.empty()) return result;
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    result.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    result.p50 = percentile(sorted, 0.50);
    result.p95 = percentile(sorted, 0.95);
    result.p99 = percentile(sorted, 0.99);
    result.minimum = sorted.front();
    result.maximum = sorted.back();
    return result;
}

static std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const char c : value) {
        if (c == '\\' || c == '"') escaped << '\\';
        escaped << c;
    }
    return escaped.str();
}

static void write_summary_json(
    const std::string& metrics_path,
    const std::string& engine_path,
    const std::string& plugin_path,
    int frames,
    int warmup_iterations,
    bool visualization_enabled,
    int fixed_track_count,
    bool collision_optimization,
    bool occupancy_dump_enabled,
    int preprocessed_image_dump_frames,
    const std::string& temporal_protocol,
    bool carry_state_across_scenes,
    const LatencySummary& model,
    const LatencySummary& inference,
    const LatencySummary& e2e) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, device);
    std::ofstream metrics(metrics_path);
    metrics << std::fixed << std::setprecision(6);
    metrics << "{\n"
            << "  \"schema_version\": 2,\n"
            << "  \"engine_path\": \"" << json_escape(engine_path) << "\",\n"
            << "  \"plugin_path\": \"" << json_escape(plugin_path) << "\",\n"
            << "  \"gpu\": \"" << json_escape(properties.name) << "\",\n"
            << "  \"tensorrt_compile_version\": \"" << NV_TENSORRT_MAJOR << "."
            << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << "\",\n"
            << "  \"frames\": " << frames << ",\n"
            << "  \"warmup_iterations\": " << warmup_iterations << ",\n"
            << "  \"visualization_enabled\": " << (visualization_enabled ? "true" : "false") << ",\n"
            << "  \"fixed_track_input_count\": " << fixed_track_count << ",\n"
            << "  \"collision_optimization_enabled\": " << (collision_optimization ? "true" : "false") << ",\n"
            << "  \"occupancy_dump_enabled\": " << (occupancy_dump_enabled ? "true" : "false") << ",\n"
            << "  \"preprocessed_image_dump_frames\": " << preprocessed_image_dump_frames << ",\n"
            << "  \"temporal_protocol\": \"" << json_escape(temporal_protocol) << "\",\n"
            << "  \"carry_state_across_scenes\": " << (carry_state_across_scenes ? "true" : "false") << ",\n"
            << "  \"collision_optimization_audit\": {\n"
            << "    \"decode_calls\": " << collision_optimization_audit().decode_calls << ",\n"
            << "    \"enabled_calls\": " << collision_optimization_audit().enabled_calls << ",\n"
            << "    \"frames_with_positive_occupancy\": " << collision_optimization_audit().frames_with_positive_occupancy << ",\n"
            << "    \"positive_occupancy_cells\": " << collision_optimization_audit().positive_occupancy_cells << ",\n"
            << "    \"frames_with_candidates\": " << collision_optimization_audit().frames_with_candidates << ",\n"
            << "    \"candidate_points\": " << collision_optimization_audit().candidate_points << ",\n"
            << "    \"frames_modified\": " << collision_optimization_audit().frames_modified << ",\n"
            << "    \"points_modified\": " << collision_optimization_audit().points_modified << ",\n"
            << "    \"max_point_delta_m\": " << collision_optimization_audit().max_point_delta_m << "\n"
            << "  },\n"
            << "  \"definitions\": {\n"
            << "    \"model_enqueue\": \"CUDA event around TensorRT enqueueV3 on the inference stream\",\n"
            << "    \"inference_call\": \"Synchronized wall time for H2D, enqueueV3, DDS handling and D2H\",\n"
            << "    \"end_to_end\": \"Wall time for metadata read, six-JPEG decode, GPU preprocessing, temporal-state update, inference call and output decoding; visualization and result-file writes excluded\"\n"
            << "  },\n";
    auto write_metric = [&metrics](const char* name, const LatencySummary& value, bool trailing_comma) {
        metrics << "  \"" << name << "\": {\"mean_ms\": " << value.mean
                << ", \"p50_ms\": " << value.p50
                << ", \"p95_ms\": " << value.p95
                << ", \"p99_ms\": " << value.p99
                << ", \"min_ms\": " << value.minimum
                << ", \"max_ms\": " << value.maximum
                << ", \"fps_from_mean\": " << (value.mean > 0.0 ? 1000.0 / value.mean : 0.0) << "}"
                << (trailing_comma ? "," : "") << "\n";
    };
    write_metric("model_enqueue", model, true);
    write_metric("inference_call", inference, true);
    write_metric("end_to_end", e2e, false);
    metrics << "}\n";
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s ENGINE PLUGIN INPUT_DIR OUTPUT_DIR NUM_FRAMES [METRICS_JSON] [WARMUP_ITERS] [VISUALIZE_0_OR_1] [FIXED_TRACK_COUNT] [official_literal|scene_reset]\n", argv[0]);
        return 2;
    }
    const std::string engine_pth = argv[1];
    const std::string plugin_pth = argv[2];
    const std::string input_pth = argv[3];
    std::string output_pth = argv[4];
    const int num_frames = std::stoi(argv[5]);
    if (output_pth.back() == '/') output_pth.pop_back();
    const std::string metrics_pth = argc > 6 ? argv[6] : output_pth + "/latency_metrics.json";
    const int num_warmup_iter = argc > 7 ? std::stoi(argv[7]) : 10;
    const bool enable_visualization = argc > 8 ? std::stoi(argv[8]) != 0 : true;
    const int fixed_track_count = argc > 9 ? std::stoi(argv[9]) : 0;
    std::string temporal_protocol = argc > 10 ? argv[10] : "";
    if (temporal_protocol.empty()) {
        const char* carry_scene_state_env = std::getenv("UNIAD_CARRY_STATE_ACROSS_SCENES");
        const bool legacy_carry = carry_scene_state_env != nullptr
            && std::string(carry_scene_state_env) != "0";
        temporal_protocol = legacy_carry ? "official_literal" : "scene_reset";
    }
    if (num_frames <= 0 || num_warmup_iter < 0 ||
        (fixed_track_count != 0 &&
         (fixed_track_count < TRACK_INS_MIN || fixed_track_count > TRACK_INS_MAX)) ||
        (temporal_protocol != "official_literal" && temporal_protocol != "scene_reset")) {
        fprintf(stderr, "[ERROR] Invalid frame, warmup, or fixed-track count argument.\n");
        return 2;
    }
    struct stat output_stat;
    if (stat(output_pth.c_str(), &output_stat) != 0 &&
        mkdir(output_pth.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
        fprintf(stderr, "[ERROR] Could not create output directory %s.\n", output_pth.c_str());
        return 2;
    }

    void* so_handle = dlopen(plugin_pth.c_str(), RTLD_NOW);
    if (so_handle == nullptr) {
        fprintf(stderr, "[ERROR] Failed to load plugin %s: %s\n", plugin_pth.c_str(), dlerror());
        return 3;
    }
    // create the inference kernel
    std::shared_ptr<UniAD::Kernel> kernel = create_kernel(engine_pth);
    if (kernel == nullptr) {
        printf("[ERROR] Failed to create kernel in the main interface.\n");
        return -1;
    }
    kernel->print_info();

    cudaStream_t stream;
    checkRuntime(cudaStreamCreate(&stream));

    std::vector<std::vector<std::string>> infos = parse_input_info(input_pth, num_frames, 0);
    if (infos.size() != static_cast<size_t>(num_frames)) {
        fprintf(stderr, "[ERROR] Requested %d frames but info.txt contains only %zu usable rows.\n", num_frames, infos.size());
        return 4;
    }
    const UniAD::KernelParams kernel_params_ref;
    int original_width, original_height, original_channel;
    auto images_dummy = load_images(infos, 0, original_width, original_height, original_channel);
    if (!valid_images(images_dummy)) {
        fprintf(stderr, "[ERROR] Failed to load all six images for frame 0.\n");
        free_images(images_dummy);
        return 4;
    }
    free_images(images_dummy);
    std::shared_ptr<ImgPreProcess> pre_processor = std::make_shared<ImgPreProcess>(
        original_width, original_height, original_channel,
        UNIAD_IMAGE_RESIZE_SCALE, 32);

    // warmup
    UniAD::KernelInput warmup_input;
    UniAD::KernelOutput dummy_output;
    std::vector<float> warmup_scene(32, 0.0f);
    bool have_warmup_scene = false;
    load_input_frame(input_pth, 0, warmup_input, warmup_scene, have_warmup_scene);
    set_fixed_track_input_shapes(warmup_input, fixed_track_count);
    auto warmup_images = load_images(infos, 0);
    warmup_input.img = pre_processor->img_pre_processing(warmup_images, stream);
    free_images(warmup_images);
    warmup_input.input_shapes["img"] = kernel_params_ref.input_max_shapes.at("img");
    relink_input(warmup_input);
    printf("[INFO] Engine warm-up start (%d iterations).\n", num_warmup_iter);
    for (int i=0; i<num_warmup_iter; ++i) {
        checkRuntime(cudaStreamSynchronize(stream));
        kernel->forward_one_frame(warmup_input, dummy_output, false, stream);
        checkRuntime(cudaStreamSynchronize(stream));
    }
    printf("[INFO] Engine warm-up done.\n");

    std::string img_dump_path = output_pth + "/dumped_video_results";
    if (enable_visualization && stat(img_dump_path.c_str(), &output_stat) != 0) {
        mkdir(img_dump_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    std::ofstream frame_metrics(metrics_pth + ".frames.csv");
    frame_metrics << "frame,scene_changed,model_enqueue_ms,inference_call_ms,end_to_end_ms,decoded_boxes,positive_occupancy_cells,collision_candidate_points,collision_modified_points,collision_max_delta_m\n";
    std::ofstream planning_predictions(output_pth + "/planning_predictions.csv");
    std::ofstream raw_planning_predictions(output_pth + "/planning_predictions_raw.csv");
    const char* dump_occupancy_env = std::getenv("UNIAD_DUMP_OCCUPANCY");
    const bool dump_occupancy = dump_occupancy_env != nullptr
        && std::string(dump_occupancy_env) != "0";
    std::ofstream occupancy_dump;
    std::vector<TRT_INT_TYPE> occupancy_shape;
    if (dump_occupancy) {
        occupancy_dump.open(output_pth + "/seg_out.packbits", std::ios::binary);
        if (!occupancy_dump) {
            fprintf(stderr, "[ERROR] Could not create packed occupancy dump.\n");
            return 2;
        }
    }
    const char* dump_images_env = std::getenv(
        "UNIAD_DUMP_PREPROCESSED_IMAGE_FRAMES");
    const int dump_image_frames = dump_images_env != nullptr
        ? std::max(0, std::atoi(dump_images_env)) : 0;
    std::ofstream preprocessed_image_dump;
    if (dump_image_frames > 0) {
        preprocessed_image_dump.open(
            output_pth + "/preprocessed_img.float32", std::ios::binary);
        if (!preprocessed_image_dump) {
            fprintf(stderr, "[ERROR] Could not create preprocessed image dump.\n");
            return 2;
        }
    }
    planning_predictions << "frame";
    raw_planning_predictions << "frame";
    for (int step = 0; step < 6; ++step) {
        planning_predictions << ",x" << step + 1 << ",y" << step + 1;
        raw_planning_predictions << ",x" << step + 1 << ",y" << step + 1;
    }
    planning_predictions << "\n";
    raw_planning_predictions << "\n";

    std::vector<double> model_samples;
    std::vector<double> inference_samples;
    std::vector<double> e2e_samples;
    model_samples.reserve(num_frames);
    inference_samples.reserve(num_frames);
    e2e_samples.reserve(num_frames);
    std::unique_ptr<UniAD::KernelOutput> previous_output;
    std::vector<float> current_scene(32, 0.0f);
    bool have_scene = false;
    const char* disable_temporal_env = std::getenv("UNIAD_DISABLE_TEMPORAL_STATE");
    const bool disable_temporal_state = disable_temporal_env != nullptr
        && std::string(disable_temporal_env) != "0";
    const bool carry_state_across_scenes = temporal_protocol == "official_literal";
    if (disable_temporal_state) {
        printf("[INFO] Temporal state propagation disabled for independent-frame benchmarking.\n");
    }
    if (carry_state_across_scenes) {
        printf("[INFO] External temporal state is carried across scene boundaries; use_prev_bev=0 requests the model-internal reset.\n");
    } else {
        printf("[INFO] External temporal state is reset at every scene boundary.\n");
    }

    for (int i=0; i<num_frames; ++i) {
        UniAD::KernelInput input;
        std::unique_ptr<UniAD::KernelOutput> output(new UniAD::KernelOutput());
        const auto e2e_begin = std::chrono::steady_clock::now();
        const bool scene_changed = load_input_frame(input_pth, i, input, current_scene, have_scene);
        set_fixed_track_input_shapes(input, fixed_track_count);
        if (disable_temporal_state) input.use_prev_bev[0] = 0;
        else if (previous_output && (!scene_changed || carry_state_across_scenes)) {
            temporal_info_assign(input, *previous_output, fixed_track_count);
        }

        auto images = load_images(infos, i);
        if (!valid_images(images)) {
            fprintf(stderr, "[ERROR] Failed to load all six images for frame %d.\n", i);
            free_images(images);
            return 5;
        }
        input.img = pre_processor->img_pre_processing(images, stream);
        if (i < dump_image_frames) {
            preprocessed_image_dump.write(
                reinterpret_cast<const char*>(input.img.data()),
                input.img.size() * sizeof(float));
        }
        input.input_shapes["img"] = kernel_params_ref.input_max_shapes.at("img");
        relink_input(input);

        checkRuntime(cudaStreamSynchronize(stream));
        const auto inference_begin = std::chrono::steady_clock::now();
        kernel->forward_one_frame(input, *output, true, stream);
        checkRuntime(cudaStreamSynchronize(stream));
        const auto inference_end = std::chrono::steady_clock::now();

        if (dump_occupancy) {
            const auto& frame_shape = output->output_shapes.at("seg_out");
            if (occupancy_shape.empty()) occupancy_shape = frame_shape;
            if (frame_shape != occupancy_shape) {
                fprintf(stderr, "[ERROR] seg_out shape changed at frame %d.\n", i);
                return 7;
            }
            std::vector<std::uint8_t> packed((output->seg_out.size() + 7) / 8, 0);
            for (std::size_t index = 0; index < output->seg_out.size(); ++index) {
                if (output->seg_out[index] != 0) {
                    packed[index / 8] |= static_cast<std::uint8_t>(1U << (index % 8));
                }
            }
            occupancy_dump.write(
                reinterpret_cast<const char*>(packed.data()), packed.size());
        }

        const std::vector<std::pair<float, float>> raw_planning =
            decode_raw_planning_traj(*output);
        const std::vector<std::pair<float, float>> planning = decode_planning_traj(*output);
        if (!finite_planning(planning)) {
            fprintf(stderr, "[ERROR] Non-finite planning trajectory at frame %d.\n", i);
            return 6;
        }
        const std::vector<std::vector<float>> boxes = decode_bbox(*output);
        const std::string command = decode_command(input);
        (void)command;
        const auto e2e_end = std::chrono::steady_clock::now();

        const double model_ms = kernel->last_model_latency_ms();
        const double inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_begin).count();
        const double e2e_ms = std::chrono::duration<double, std::milli>(e2e_end - e2e_begin).count();
        model_samples.push_back(model_ms);
        inference_samples.push_back(inference_ms);
        e2e_samples.push_back(e2e_ms);
        frame_metrics << i << "," << (scene_changed ? 1 : 0) << ","
                      << std::fixed << std::setprecision(6) << model_ms << ","
                      << inference_ms << "," << e2e_ms << "," << boxes.size() << ","
                      << collision_optimization_audit().last_positive_occupancy_cells << ","
                      << collision_optimization_audit().last_candidate_points << ","
                      << collision_optimization_audit().last_points_modified << ","
                      << collision_optimization_audit().last_max_point_delta_m << "\n";
        planning_predictions << i;
        for (const auto& point : planning) planning_predictions << "," << point.first << "," << point.second;
        planning_predictions << "\n";
        raw_planning_predictions << i;
        for (const auto& point : raw_planning) {
            raw_planning_predictions << "," << point.first << "," << point.second;
        }
        raw_planning_predictions << "\n";

        if (enable_visualization) {
            visualize(images, input, *output, planning,
                      img_dump_path + "/" + std::to_string(i) + ".jpg", stream);
        }
        free_images(images);
        if (!disable_temporal_state) previous_output = std::move(output);
        printf("[INFO] Frame %d/%d: enqueue %.3f ms, inference %.3f ms, end-to-end %.3f ms.\n",
               i + 1, num_frames, model_ms, inference_ms, e2e_ms);
    }
    checkRuntime(cudaStreamSynchronize(stream));

    const LatencySummary model_summary = summarize(model_samples);
    const LatencySummary inference_summary = summarize(inference_samples);
    const LatencySummary e2e_summary = summarize(e2e_samples);
    write_summary_json(metrics_pth, engine_pth, plugin_pth, num_frames, num_warmup_iter,
                       enable_visualization, fixed_track_count,
                       collision_optimization_enabled(), dump_occupancy,
                       std::min(num_frames, dump_image_frames),
                       temporal_protocol,
                       carry_state_across_scenes,
                       model_summary, inference_summary, e2e_summary);
    if (dump_occupancy) {
        occupancy_dump.close();
        std::ofstream occupancy_manifest(output_pth + "/seg_out.packbits.manifest.json");
        occupancy_manifest << "{\n"
                           << "  \"schema_version\": 1,\n"
                           << "  \"packing\": \"numpy packbits compatible, little bit order\",\n"
                           << "  \"temporal_protocol\": \"" << json_escape(temporal_protocol) << "\",\n"
                           << "  \"frames\": " << num_frames << ",\n"
                           << "  \"shape\": [";
        for (std::size_t index = 0; index < occupancy_shape.size(); ++index) {
            if (index) occupancy_manifest << ", ";
            occupancy_manifest << occupancy_shape[index];
        }
        occupancy_manifest << "]\n}\n";
    }
    if (dump_image_frames > 0) {
        preprocessed_image_dump.close();
        std::ofstream image_manifest(
            output_pth + "/preprocessed_img.float32.manifest.json");
        image_manifest << "{\n"
                       << "  \"schema_version\": 1,\n"
                       << "  \"dtype\": \"float32\",\n"
                       << "  \"layout\": \"NCHW\",\n"
                       << "  \"frames\": " << std::min(num_frames, dump_image_frames) << ",\n"
                       << "  \"shape_per_frame\": [1, 6, 3, "
                       << UNIAD_IMG_H << ", " << UNIAD_IMG_W << "]\n"
                       << "}\n";
    }
    for (const auto& prediction_file : {
            std::make_pair(output_pth + "/planning_predictions.csv", "collision-optimized"),
            std::make_pair(output_pth + "/planning_predictions_raw.csv", "raw outs_planning")}) {
        std::ofstream manifest(prediction_file.first + ".manifest.json");
        manifest << "{\n"
                 << "  \"schema_version\": 1,\n"
                 << "  \"producer\": \"UniAD TensorRT enqueueV3 runtime\",\n"
                 << "  \"temporal_protocol\": \"" << json_escape(temporal_protocol) << "\",\n"
                 << "  \"frames\": " << num_frames << ",\n"
                 << "  \"trajectory\": \"" << prediction_file.second << "\"\n"
                 << "}\n";
    }
    printf("[RESULT] model enqueue: mean %.3f ms, p50 %.3f ms, FPS %.3f.\n",
           model_summary.mean, model_summary.p50, 1000.0 / model_summary.mean);
    printf("[RESULT] inference call: mean %.3f ms, p50 %.3f ms.\n",
           inference_summary.mean, inference_summary.p50);
    printf("[RESULT] end-to-end: mean %.3f ms, p50 %.3f ms, FPS %.3f.\n",
           e2e_summary.mean, e2e_summary.p50, 1000.0 / e2e_summary.mean);
    printf("[RESULT] Metrics written to %s.\n", metrics_pth.c_str());
    if (enable_visualization) printf("[INFO] Visualization results written to %s.\n", img_dump_path.c_str());
    checkRuntime(cudaStreamDestroy(stream));
    previous_output.reset();
    kernel.reset();
    pre_processor.reset();
    dlclose(so_handle);
    return 0;
}
