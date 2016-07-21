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
// AlexNet with weights & biases from file
void alexnet(uint32_t batch_size, std::string img_dir)
{
    using namespace neural;
    auto input  = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ batch_size,{ 227, 227 }, 3, } });
    auto output = memory::allocate({ engine::reference, memory::format::xb_f32,{ batch_size,{ 1000 }} });
    auto img_list = get_directory_images(img_dir);
    if (img_list.empty())
        throw std::runtime_error("Specified path doesn't contain image data\n");
    auto images_list_iterator = img_list.begin();
    auto images_list_end = img_list.end();
    auto number_of_batches = (img_list.size() % batch_size == 0) 
        ? img_list.size() / batch_size : img_list.size() / batch_size + 1;

    for (auto batch = 0; batch < number_of_batches; batch++)
    {   // TODO: run network in loop
        while (images_list_iterator != images_list_end)
        {
            // TODO: pack into bachtes
            std::cout << images_list_iterator->c_str();
            images_list_iterator++;
        }
    }

    // TODO: move this to separate function;
    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    auto conv1 = convolution::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        { 
            input,
            file::create({ engine::reference, "conv1_weights.nnd" }),
            file::create({ engine::reference, "conv1_biases.nnd" })
        },
        {0,{ 0, 0}, 0},
        { 1, { 4, 4 }, 1 },
        padding::zero });

    auto relu1 = relu::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        conv1
    });

    auto lrn1 = normalization::response::create(
    {
        engine::reference,
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
        engine::reference,
        pooling::mode::max,
        memory::format::yxfb_f32,
        lrn1,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto conv2_group2 = convolution::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        {
            pool1,
            file::create({ engine::cpu, "conv2_g1_weights.nnd" }),
            file::create({ engine::cpu, "conv2_g1_biases.nnd" }),
            file::create({ engine::cpu, "conv2_g2_weights.nnd" }),
            file::create({ engine::cpu, "conv2_g2_biases.nnd" }),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        2
    });

    auto relu2 = relu::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        conv2_group2
    });

    auto lrn2 = normalization::response::create(
    {
        engine::reference,
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
        engine::reference,
        pooling::mode::max,
        memory::format::yxfb_f32,
        lrn2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto conv3 = convolution::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        {
            pool2,
            file::create({ engine::cpu, "conv3_weights.nnd" }),
            file::create({ engine::cpu, "conv3_biases.nnd" }),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero
    });

    auto relu3 = relu::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        conv3
    });

    auto conv4_group2 = convolution::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        {
            relu3,
            file::create({ engine::cpu, "conv4_g1_weights.nnd" }),
            file::create({ engine::cpu, "conv4_g1_biases.nnd" }),
            file::create({ engine::cpu, "conv4_g2_weights.nnd" }),
            file::create({ engine::cpu, "conv4_g2_biases.nnd" }),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        2
    });

    auto relu4 = relu::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        conv4_group2
    });

    auto conv5_group2 = convolution::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        {
            relu4,
            file::create({ engine::cpu, "conv5_g1_weights.nnd" }),
            file::create({ engine::cpu, "conv5_g1_biases.nnd" }),
            file::create({ engine::cpu, "conv5_g2_weights.nnd" }),
            file::create({ engine::cpu, "conv5_g2_biases.nnd" }),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        2
    });

    auto relu5 = relu::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        conv5_group2
    });

    auto pool5 = pooling::create(
    {
        engine::reference,
        pooling::mode::max,
        memory::format::yxfb_f32,
        relu5,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto fc6 = fully_connected_relu::create(
    {
        engine::reference,
        memory::format::xb_f32,
        pool5,
        file::create({ engine::cpu, "fc6_weights.nnd" }),
        file::create({ engine::cpu, "fc6_biases.nnd" }),
        0
    });

    auto fc7 = fully_connected_relu::create(
    {
        engine::reference,
        memory::format::xb_f32,
        fc6,
        file::create({ engine::cpu, "fc7_weights.nnd" }),
        file::create({ engine::cpu, "fc7_biases.nnd" }),
        0
    });

    auto fc8 = fully_connected_relu::create(
    {
        engine::reference,
        memory::format::xb_f32,
        fc7,
        file::create({ engine::cpu, "fc8_weights.nnd" }),
        file::create({ engine::cpu, "fc8_biases.nnd" }),
        0
    });

    auto softmax = normalization::softmax::create(
    {
        engine::reference,
        output,
        fc8
    });
}
