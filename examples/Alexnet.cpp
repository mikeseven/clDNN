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
#include <iostream>
#include <string>

using namespace neural;
// AlexNet with weights & biases from file
std::chrono::high_resolution_clock::duration execute_alexnet(primitive& input, primitive& output, engine::type eng, bool dump_hl)
{
    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    instrumentation::timer timer_build, timer_execution;
    std::cout << "Building Alexnet started" << std::endl;
    timer_build.start();
    auto mean = mean_subtract::create(
    {
        eng,
        memory::format::yxfb_f32,
        input,
        file::create({eng,"imagenet_mean.nnd"})
    });

    auto conv1 = convolution::create(
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

    auto relu1 = relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        conv1
    });

    auto lrn1 = normalization::response::create(
    {
        eng,
        memory::format::yxfb_f32,
        relu1,
        5,
        padding::zero,
        1.0f,
        0.00002f,
        0.75f
    });

    auto pool1 = pooling::create(
    {
        eng,
        pooling::mode::max,
        memory::format::yxfb_f32,
        lrn1,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto conv2_group2 = convolution::create(
    {
        eng,
        memory::format::yxfb_f32,
        {
            pool1,
            file::create({ eng, "conv2_g1_weights.nnd" }),
            file::create({ eng, "conv2_g1_biases.nnd" }),
            file::create({ eng, "conv2_g2_weights.nnd" }),
            file::create({ eng, "conv2_g2_biases.nnd" }),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        2
    });

    auto relu2 = relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        conv2_group2
    });

    auto lrn2 = normalization::response::create(
    {
        eng,
        memory::format::yxfb_f32,
        relu2,
        5,
        padding::zero,
        1.0f,
        0.0001f,
        0.75
    });

    auto pool2 = pooling::create(
    {
        eng,
        pooling::mode::max,
        memory::format::yxfb_f32,
        lrn2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto conv3 = convolution::create(
    {
        eng,
        memory::format::yxfb_f32,
        {
            pool2,
            file::create({ eng, "conv3_weights.nnd" }),
            file::create({ eng, "conv3_biases.nnd" }),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero
    });

    auto relu3 = relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        conv3
    });

    auto conv4_group2 = convolution::create(
    {
        eng,
        memory::format::yxfb_f32,
        {
            relu3,
            file::create({ eng, "conv4_g1_weights.nnd" }),
            file::create({ eng, "conv4_g1_biases.nnd" }),
            file::create({ eng, "conv4_g2_weights.nnd" }),
            file::create({ eng, "conv4_g2_biases.nnd" }),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        2
    });

    auto relu4 = relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        conv4_group2
    });

    auto conv5_group2 = convolution::create(
    {
        eng,
        memory::format::yxfb_f32,
        {
            relu4,
            file::create({ eng, "conv5_g1_weights.nnd" }),
            file::create({ eng, "conv5_g1_biases.nnd" }),
            file::create({ eng, "conv5_g2_weights.nnd" }),
            file::create({ eng, "conv5_g2_biases.nnd" }),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        2
    });

    auto relu5 = relu::create(
    {
        eng,
        memory::format::yxfb_f32,
        conv5_group2
    });

    auto pool5 = pooling::create(
    {
        eng,
        pooling::mode::max,
        memory::format::yxfb_f32,
        relu5,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto fc6 = fully_connected_relu::create(
    {
        eng,
        memory::format::xb_f32,
        pool5,
        file::create({ eng, "fc6_weights.nnd", file::weights_type::fully_connected}),
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

    timer_build.stop();
    std::cout << "Building Alexnet finished in " << timer_build.time_diff_string() << std::endl;
    std::cout << "Start execution" << std::endl;

    timer_execution.start();
    execute({
        mean, //mean
        conv1,relu1,lrn1,pool1, //stage 0
        conv2_group2,relu2,lrn2, pool2,
        conv3,relu3,
        conv4_group2, relu4,
        conv5_group2, relu5, pool5,
        fc6,
        fc7,
        fc8,
        softmax,output }).wait();
    timer_execution.stop();
    std::cout << "Alexnet execution finished in " << timer_execution.time_diff_string() << std::endl;
    //instrumentation::log_memory_to_file(conv1.output[0],"conv1");
    if (dump_hl)
    {
        instrumentation::logger::log_memory_to_file(input, "input0");
        instrumentation::logger::log_memory_to_file(mean, "mean");
        instrumentation::logger::log_memory_to_file(conv1.output[0], "conv1");
        instrumentation::logger::log_memory_to_file(relu1.output[0], "relu1");
        instrumentation::logger::log_memory_to_file(lrn1.output[0], "lrn1");
        instrumentation::logger::log_memory_to_file(pool1.output[0], "pool1");
        instrumentation::logger::log_memory_to_file(conv2_group2.output[0], "conv2_group2");
        instrumentation::logger::log_memory_to_file(relu2.output[0], "relu2");
        instrumentation::logger::log_memory_to_file(pool2.output[0], "pool2");
        instrumentation::logger::log_memory_to_file(conv3.output[0], "conv3");
        instrumentation::logger::log_memory_to_file(relu3.output[0], "relu3");
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
    return timer_execution.get_time_diff();
}

void alexnet(uint32_t batch_size, std::string img_dir, engine::type eng, bool dump_hl)
{
    auto input  = memory::allocate({ engine::reference, memory::format::byxf_f32,{ batch_size,{ 227, 227 }, 3, } });
    auto output = memory::allocate({ engine::reference, memory::format::xb_f32,{ batch_size,{ 1000 }} });
    auto img_list = get_directory_images(img_dir);
    if (img_list.empty())
        throw std::runtime_error("Specified path doesn't contain image data\n");
    auto images_list_iterator = img_list.begin();
    auto images_list_end = img_list.end();
    auto number_of_batches = (img_list.size() % batch_size == 0) 
        ? img_list.size() / batch_size : img_list.size() / batch_size + 1;
    std::vector<std::string> image_in_batches;
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
        auto ratio =  std::chrono::duration_cast<std::chrono::milliseconds>(time);
        std::cout << "Frames per second:" << (double)batch_size*1000.0/(double)ratio.count() << std::endl;
    }    
}
