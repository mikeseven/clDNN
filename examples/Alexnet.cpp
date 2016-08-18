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

// AlexNet with weights & biases from file
std::chrono::nanoseconds execute_alexnet(primitive& input, primitive& output, engine::type eng, bool dump_hl)
{
    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    std::cout << "Building Alexnet started" << std::endl;
    instrumentation::timer<> timer_build;
    auto mean = mean_subtract::create(
    {
        engine::type::reference,
        memory::format::yxfb_f32,
        input,
        file::create({ eng,"imagenet_mean.nnd" })
    });

    auto conv1 = convolution_relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        {
            mean,
            file::create({ eng, "conv1_weights.nnd" }),
            file::create({ eng, "conv1_biases.nnd" })
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 4, 4 }, 1 },
        padding::zero });

    auto pool1 = pooling::create(
    {
        eng,
        pooling::mode::max,
        memory::format::yxfb_f32,
        conv1,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto lrn1 = normalization::response::create(
    {
        eng,
        memory::format::yxfb_f32,
        pool1,
        5,
        padding::zero,
        1.0f,
        0.00002f,
        0.75f
    });

    auto conv2_group2 = convolution_relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        {
            lrn1,
            file::create({ eng, "conv2_g1_weights.nnd" }),
            file::create({ eng, "conv2_g1_biases.nnd" }),
            file::create({ eng, "conv2_g2_weights.nnd" }),
            file::create({ eng, "conv2_g2_biases.nnd" }),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        0, // negative slope for RELU
        2
    });

    auto pool2 = pooling::create(
    {
        eng,
        pooling::mode::max,
        memory::format::yxfb_f32,
        conv2_group2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto lrn2 = normalization::response::create(
    {
        eng,
        memory::format::yxfb_f32,
        pool2,
        5,
        padding::zero,
        1.0f,
        0.00002f,
        0.75
    });

    auto conv3 = convolution_relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        {
            lrn2,
            file::create({ eng, "conv3_weights.nnd" }),
            file::create({ eng, "conv3_biases.nnd" }),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero
    });

    auto conv4_group2 = convolution_relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        {
            conv3,
            file::create({ eng, "conv4_g1_weights.nnd" }),
            file::create({ eng, "conv4_g1_biases.nnd" }),
            file::create({ eng, "conv4_g2_weights.nnd" }),
            file::create({ eng, "conv4_g2_biases.nnd" }),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        0, // negative slope for RELU
        2
    });

    auto conv5_group2 = convolution_relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        {
            conv4_group2,
            file::create({ eng, "conv5_g1_weights.nnd" }),
            file::create({ eng, "conv5_g1_biases.nnd" }),
            file::create({ eng, "conv5_g2_weights.nnd" }),
            file::create({ eng, "conv5_g2_biases.nnd" }),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        0, // negative slope for RELU
        2
    });

    auto pool5 = pooling::create(
    {
        eng,
        pooling::mode::max,
        memory::format::yxfb_f32,
        conv5_group2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto fc6 = fully_connected_relu::create(
    {
        eng,
        memory::format::xb_f32,
        pool5,
        file::create({ eng, "fc6_weights.nnd", file::weights_type::fully_connected }),
        file::create({ eng, "fc6_biases.nnd" }),
        0
    });

    auto fc7 = fully_connected_relu::create(
    {
        eng,
        memory::format::xb_f32,
        fc6,
        file::create({ eng, "fc7_weights.nnd", file::weights_type::fully_connected }),
        file::create({ eng, "fc7_biases.nnd" }),
        0
    });

    auto fc8 = fully_connected_relu::create(
    {
        eng,
        memory::format::xb_f32,
        fc7,
        file::create({ eng, "fc8_weights.nnd", file::weights_type::fully_connected }),
        file::create({ eng, "fc8_biases.nnd" }),
        0
    });

    auto softmax = normalization::softmax::create(
    {
        eng,
        output,
        fc8
    });

    auto build_time = timer_build.uptime();
    std::cout << "Building Alexnet finished in " << instrumentation::to_string(build_time) << std::endl;


    std::vector<worker> workers;

    switch(eng) 
    {
    case engine::gpu:
    {
        std::cout << "GPU Program compilation started" << std::endl;
        instrumentation::timer<> timer_compilation;
        workers.push_back(worker_gpu::create({true}));
        auto compile_time = timer_compilation.uptime();
        std::cout << "GPU Program compilation finished in " << instrumentation::to_string(compile_time) << std::endl;
    }
        break;
    default:
        workers.push_back(worker_cpu::create({}));
    }

    std::cout << "Start execution" << std::endl;
    instrumentation::timer<> timer_execution;
    execute({
        mean, //mean
        conv1, pool1, lrn1, //stage 0
        conv2_group2, pool2, lrn2,
        conv3,
        conv4_group2,
        conv5_group2, pool5,
        fc6,
        fc7,
        fc8,
        softmax,output }, workers).wait();

    auto execution_time(timer_execution.uptime());
    std::cout << "Alexnet execution finished in " << instrumentation::to_string(execution_time) << std::endl;
    //instrumentation::log_memory_to_file(conv1.output[0],"conv1");
    if (dump_hl)
    {
        instrumentation::logger::log_memory_to_file(input, "input0");
        instrumentation::logger::log_memory_to_file(mean, "mean");
        instrumentation::logger::log_memory_to_file(conv1.output[0], "conv1");
        instrumentation::logger::log_memory_to_file(lrn1.output[0], "lrn1");
        instrumentation::logger::log_memory_to_file(pool1.output[0], "pool1");
        instrumentation::logger::log_memory_to_file(conv2_group2.output[0], "conv2_group2");
        instrumentation::logger::log_memory_to_file(pool2.output[0], "pool2");
        instrumentation::logger::log_memory_to_file(lrn2.output[0], "lrn2");
        instrumentation::logger::log_memory_to_file(conv3.output[0], "conv3");
        instrumentation::logger::log_memory_to_file(conv4_group2.output[0], "conv4_group2");
        instrumentation::logger::log_memory_to_file(conv5_group2.output[0], "conv5_group2");
        instrumentation::logger::log_memory_to_file(pool5.output[0], "pool5");
        instrumentation::logger::log_memory_to_file(fc6.output[0], "fc6");
        instrumentation::logger::log_memory_to_file(fc8.output[0], "fc8");
        instrumentation::logger::log_memory_to_file(softmax.output[0], "softmax");
        // for now its enought. rest wil be done when we have equals those values
    }
    else
    {
        instrumentation::logger::log_memory_to_file(output, "final_result");
    }

    if (eng == engine::gpu) {
        auto profiling_info = workers[0].as<worker_gpu&>().get_profiling_info();
        if (profiling_info.size() > 0) {
            auto max_len_it = std::max_element(std::begin(profiling_info), std::end(profiling_info), [](decltype(profiling_info)::value_type& a, decltype(profiling_info)::value_type& b) {return a.first.length() < b.first.length(); });
            std::cout << "Kernels profiling info: " << std::endl;
            auto max_len = max_len_it->first.length();
            for (auto& pi : profiling_info) {
                std::cout << std::setw(max_len) << std::left << pi.first << " " << instrumentation::to_string(pi.second) << std::endl;
            }
        }
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(execution_time);
}

void alexnet(uint32_t batch_size, std::string img_dir, engine::type eng, bool dump_hl)
{
    auto input = memory::allocate({ engine::reference, memory::format::byxf_f32,{ batch_size,{ 227, 227 }, 3, } });
    auto output = memory::allocate({ engine::reference, memory::format::xb_f32,{ batch_size,{ 1000 } } });
    auto img_list = get_directory_images(img_dir);
    if (img_list.empty())
        throw std::runtime_error("Specified path doesn't contain image data\n");
    auto images_list_iterator = img_list.begin();
    auto images_list_end = img_list.end();
    auto number_of_batches = (img_list.size() % batch_size == 0)
        ? img_list.size() / batch_size : img_list.size() / batch_size + 1;
    std::vector<std::string> image_in_batches;
    html output_file("alexnet", "alexnet run");
    for (decltype(number_of_batches) batch = 0; batch < number_of_batches; batch++)
    {
        image_in_batches.clear();
        for (uint32_t i = 0; i < batch_size && images_list_iterator != images_list_end; i++, images_list_iterator++)
            image_in_batches.push_back(*images_list_iterator);
        // load croped and resized images into input
        load_images_from_file_list(image_in_batches, input);

        // create conversion to yxfb format
        auto reordered_input = reorder::create(
        {
            engine::reference,
            memory::format::yxfb_f32,
            input.as<const memory&>().argument.size, // do not resize
            input
        });
        // reorder data
        execute({ reordered_input }).wait();
        auto time = execute_alexnet(reordered_input, output, eng, dump_hl);
        auto time_in_sec = std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(time).count();
        if(time_in_sec != 0.0)
            std::cout << "Frames per second:" << (double)batch_size / time_in_sec << std::endl;
		output_file.batch(output.as<const neural::memory&>( ), "names.txt", image_in_batches);
    }    
}
