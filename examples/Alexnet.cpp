/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/



#include "common/common_tools.h"
#include "output_parser.h"
#include <iostream>
#include <string>
#include "api/instrumentation.h"

using namespace neural;

// Building AlexNet network with loading weights & biases from file
std::vector<std::pair<primitive, std::string>> build_alexnet(const primitive& input, const primitive& output, const std::string& weights_dir, weights_optimizer& wo, bool use_half)
{
    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    std::cout << "Building Alexnet started" << std::endl;
    instrumentation::timer<> timer_build;

    auto mem_format = use_half ? memory::format::yxfb_f16 : memory::format::yxfb_f32;
    auto fc_mem_format = use_half ? memory::format::xb_f16 : memory::format::xb_f32;

    // create conversion to yxfb format and subtract mean values
    auto reordered_input = reorder::create(
    {
        mem_format,
        input.as<const memory&>().argument.size,
        input,
        wo.create_weights_from_file(join_path(weights_dir, "imagenet_mean.nnd"), file::mean)
    });

    auto conv1 = convolution::create(
    {
        mem_format,
        {
            reordered_input,
            wo.create_weights_from_file(join_path(weights_dir, "conv1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv1_biases.nnd"),  file::bias)
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 4, 4 }, 1 },
        padding::zero,
        1,
        true });

    auto pool1 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv1,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto lrn1 = normalization::response::create(
    {
        mem_format,
        pool1,
        5,
        padding::zero,
        1.0f,
        0.00002f,
        0.75f
    });

    auto conv2_group2 = convolution::create(
    {
        mem_format,
        {
            lrn1,
            wo.create_weights_from_file(join_path(weights_dir, "conv2_g1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv2_g1_biases.nnd"),  file::bias),
            wo.create_weights_from_file(join_path(weights_dir, "conv2_g2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv2_g2_biases.nnd"),  file::bias),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        2,
        true,
        0 // negative slope for RELU
    });

    auto pool2 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv2_group2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto lrn2 = normalization::response::create(
    {
        mem_format,
        pool2,
        5,
        padding::zero,
        1.0f,
        0.00002f,
        0.75
    });

    auto conv3 = convolution::create(
    {
        mem_format,
        {
            lrn2,
            wo.create_weights_from_file(join_path(weights_dir, "conv3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv3_biases.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto conv4_group2 = convolution::create(
    {
        mem_format,
        {
            conv3,
            wo.create_weights_from_file(join_path(weights_dir, "conv4_g1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv4_g1_biases.nnd"),  file::bias),
            wo.create_weights_from_file(join_path(weights_dir, "conv4_g2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv4_g2_biases.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        2,
        true,
        0 // negative slope for RELU
    });

    auto conv5_group2 = convolution::create(
    {
        mem_format,
        {
            conv4_group2,
            wo.create_weights_from_file(join_path(weights_dir, "conv5_g1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv5_g1_biases.nnd"),  file::bias),
            wo.create_weights_from_file(join_path(weights_dir, "conv5_g2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv5_g2_biases.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        2,
        true,
        0 // negative slope for RELU
    });

    auto pool5 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv5_group2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto fc6 = fully_connected::create(
    {
        fc_mem_format,
        pool5,
        wo.create_weights_from_file(join_path(weights_dir, "fc6_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc6_biases.nnd"),  file::bias),
        true,
        0
    });

    auto fc7 = fully_connected::create(
    {
        fc_mem_format,
        fc6,
        wo.create_weights_from_file(join_path(weights_dir, "fc7_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc7_biases.nnd"),  file::bias),
        true,
        0
    });

    auto fc8 = fully_connected::create(
    {
        fc_mem_format,
        fc7,
        wo.create_weights_from_file(join_path(weights_dir, "fc8_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc8_biases.nnd"),  file::bias),
        true,
        0
    });

    auto softmax = normalization::softmax::create(
    {
        output,
        fc8
    });

    auto build_time = timer_build.uptime();
    std::cout << "Building Alexnet finished in " << instrumentation::to_string(build_time) << std::endl;

    return std::vector<std::pair<primitive, std::string>> {
        { reordered_input, "reorder"},
        { conv1, "conv1" },
        { pool1, "pool1" },
        { lrn1, "lrn1" },
        { conv2_group2, "conv2_group2" },
        { pool2, "pool2" },
        { lrn2, "lrn2" },
        { conv3, "conv3" },
        { conv4_group2, "conv4_gorup2" },
        { conv5_group2, "conv5_group2" },
        { pool5, "pool5" },
        { fc6, "fc6" },
        { fc7, "fc7" },
        { fc8, "fc8" },
        { softmax, "softmax" }
    };
}




void alexnet(const execution_params &ep)
{
    uint32_t batch_size = ep.batch;

    uint32_t gpu_batch_size = get_gpu_batch_size(batch_size);
    if (gpu_batch_size != batch_size)
    {
        std::cout << "WARNING: This is not the optimal batch size. You have " << (gpu_batch_size - batch_size)
            << " dummy images per batch!!! Please use batch=" << gpu_batch_size << "." << std::endl;
    }
    gpu::configuration::get().enable_profiling = ep.profiling;

    auto img_list = get_directory_images(ep.input_dir);
    if (img_list.empty())
        throw std::runtime_error("specified input images directory is empty (does not contain image data)");

    auto number_of_batches = (img_list.size() % batch_size == 0)
        ? img_list.size() / batch_size : img_list.size() / batch_size + 1;

    html output_file("alexnet", "alexnet run");

    weights_optimizer weights_optimizer(ep.optimize_weights, ep.use_half);

    auto input = memory::allocate({ep.use_half ? memory::format::byxf_f16 : memory::format::byxf_f32, {gpu_batch_size, {227, 227}, 3}});
    auto output = memory::allocate({ep.use_half ? memory::format::xb_f16 : memory::format::xb_f32, {gpu_batch_size, {1000}}});

    // build alexnet
    std::vector<std::pair<primitive, std::string>> primitives = build_alexnet(input, output, ep.weights_dir, weights_optimizer, ep.use_half);

    // create worker
    worker worker = create_worker();

    // optimize weights if needed
    if (ep.optimize_weights)
    {
        weight_optimization(weights_optimizer, worker);
    }

    std::vector<std::string> images_in_batch;
    auto images_list_iterator = img_list.begin();
    auto images_list_end = img_list.end();
    for (decltype(number_of_batches) batch = 0; batch < number_of_batches; batch++)
    {
        images_in_batch.clear();
        for (uint32_t i = 0; i < batch_size && images_list_iterator != images_list_end; ++i, ++images_list_iterator)
        {
            images_in_batch.push_back(*images_list_iterator);
        }

        // load croped and resized images into input
        if (ep.use_half)
        {
            load_images_from_file_list<half_t>(images_in_batch, input);
        }
        else
        {
            load_images_from_file_list(images_in_batch, input);
        }

        // execute alexnet
        auto time = execute_topology(worker, primitives, output, ep, "alexnet", 15);

        auto time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
        output_file.batch(output.as<const neural::memory&>(), join_path(get_executable_info()->dir(), "names.txt"), images_in_batch);
        if (time_in_sec != 0.0)
        {
            std::cout << "Frames per second:" << (double)batch_size / time_in_sec << std::endl;
        }
    }
}

