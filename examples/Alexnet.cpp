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

#if 1
#include "api/neural.h"
#include "common/common_tools.h"
#include <iostream>
#include <string>
// AlexNet with weights & biases from file
void Alexnet(uint32_t batch_size, std::string img_dir)
{
    using namespace neural;

    //char  *input_buffer = nullptr;
    //char *output_buffer = nullptr;
    auto img_list = get_directory_images(img_dir);
    auto images_list_iterator = img_list.begin();
    auto images_list_end = img_list.end();

    while (images_list_iterator != images_list_end)
    {
        std::cout << images_list_iterator->c_str();
        images_list_iterator++;
    }
    auto input = memory::describe({ engine::reference, memory::format::yxfb_f32,{ batch_size,{ 227, 227 }, 3, } });
    // conv1->pool1
    //auto hidden1 = memory::describe({ engine::reference, memory::format::yxfb_f32,{ batch_size,{ 224, 224 }, 3, } })
    auto output = memory::describe({ engine::reference, memory::format::xb_f32,{ batch_size,{ { 1000 } },1 } });

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
    auto conv2g1 = convolution::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        {
            pool1,
            file::create({engine::cpu, "conv2_g1_weights.nnd"}),
            file::create({engine::cpu, "conv2_g1_biases.nnd"}),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero
    });

    auto conv2g2 = convolution::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        {
            pool1,
            file::create({ engine::cpu, "conv2_g1_weights.nnd" }),
            file::create({ engine::cpu, "conv2_g1_biases.nnd" }),
        },
        { 0,{ -2, -2 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero
    });


    auto relu2g1 = relu::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        conv2g1
    });

    auto relu2g2 = relu::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        conv2g2
    });

    auto lrn2g1 = normalization::response::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        relu2g1,
        5,
        padding::zero,
        1.0f,
        0.0001f,
        0.75
    });

    auto lrn2g2 = normalization::response::create(
    {
        engine::reference,
        memory::format::yxfb_f32,
        relu2g2,
        5,
        padding::zero,
        1.0f,
        0.0001f,
        0.75
    });

    auto pool2g1 = pooling::create(
    {
        engine::reference,
        pooling::mode::max,
        memory::format::yxfb_f32,
        lrn2g1,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto pool2g2 = pooling::create(
    {
        engine::reference,
        pooling::mode::max,
        memory::format::yxfb_f32,
        lrn2g2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    // TODO: place here primitive:: memory that will merge two outputs into one 

     /* auto pool2      = pooling::create({engine::cpu, pooling::mode::max, memory::format::yxfb_f32, conv_relu2, 3, 2, padding::zero});
    auto lrn2       = normalization::response::create({engine::cpu, memory::format::yxfb_f32, pool2, 5, padding::zero, 1.0f, 0.00002f, 0.75f });
    auto conv_relu3 = convolution_relu::create({engine::cpu, memory::format::yxfb_f32, lrn2, file::create({engine::cpu, "weight3.nnb"}), file::create({engine::cpu, "bias3.nnb"}), padding::zero, 0.0f});
    auto conv_relu4 = convolution_relu::create({engine::cpu, memory::format::yxfb_f32, conv_relu3, file::create({engine::cpu, "weight4.nnb"}), file::create({engine::cpu, "bias4.nnb"}), padding::zero, 0.0f});
    auto conv_relu5 = convolution_relu::create({engine::cpu, memory::format::yxfb_f32, conv_relu4, file::create({engine::cpu, "weight5.nnb"}), file::create({engine::cpu, "bias5.nnb"}), padding::zero, 0.0f});
    auto pool5      = pooling::create({engine::cpu, pooling::mode::max, memory::format::yxfb_f32, conv_relu5, 3, 2, padding::zero});
    auto fc_relu6   = fully_connected_relu::create({engine::cpu, memory::format::yxfb_f32, pool5, file::create({engine::cpu, "weight6.nnb"}), file::create({engine::cpu, "bias6.nnb"}), 0.0f});
    auto fc_relu7   = fully_connected_relu::create({engine::cpu, memory::format::yxfb_f32, fc_relu6, file::create({engine::cpu, "weight7.nnb"}), file::create({engine::cpu, "bias7.nnb"}), 0.0f});
    auto fc_relu8    = fully_connected_relu::create({engine::cpu, memory::format::yxfb_f32, fc_relu7, file::create({engine::cpu, "weight8.nnb"}), file::create({engine::cpu, "bias8.nnb"}), 0.0f});
    auto soft_max   = normalization::softmax::create({engine::cpu, output, fc_relu8});

    execute({input(input_buffer), output(input_buffer), conv_relu1, pool1, lrn1, conv_relu2, pool2, lrn2, conv_relu3, conv_relu4, conv_relu5, pool5, fc_relu6, fc_relu7, fc_relu8, soft_max}).wait();*/
}
#endif