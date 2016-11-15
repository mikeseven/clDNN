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
#include <string>

using namespace neural;

// Building vgg16 network with loading weights & biases from file
std::vector<std::pair<primitive, std::string>> build_vgg16(const primitive& input, const primitive& output, const std::string& weights_dir, weights_optimizer& wo, bool use_half)
{
    auto mem_format = use_half ? memory::format::yxfb_f16 : memory::format::yxfb_f32;
    auto fc_mem_format = use_half ? memory::format::xb_f16 : memory::format::xb_f32;

    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]

    // create conversion to yxfb format and subtract mean values
    auto reordered_input = reorder::create(
    {
        mem_format,
        input.as<const memory&>().argument.size,
        input,
        wo.create_weights_from_file(join_path(weights_dir, "imagenet_mean.nnd"), file::mean)
    });

    auto conv1_1 = convolution::create(
    {
        mem_format,
        {
            reordered_input,
            wo.create_weights_from_file(join_path(weights_dir, "conv1_1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv1_1_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true } );

    auto conv1_2 = convolution::create(
    {
        mem_format,
        {
            conv1_1,
            wo.create_weights_from_file(join_path(weights_dir, "conv1_2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv1_2_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true });


    auto pool1 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv1_2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 2,2 },1 }, // kernel
        padding::zero
    });

    auto conv2_1 = convolution::create(
    {
        mem_format,
        {
            pool1,
            wo.create_weights_from_file(join_path(weights_dir, "conv2_1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv2_1_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true, // negative slope for RELU
    });

    auto conv2_2 = convolution::create(
    {
        mem_format,
        {
            conv2_1,
            wo.create_weights_from_file(join_path(weights_dir, "conv2_2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv2_2_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true, // negative slope for RELU
    });

    auto pool2 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv2_2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 2,2 },1 }, // kernel
        padding::zero
    });

    auto conv3_1 = convolution::create(
    {
        mem_format,
        {
            pool2,
            wo.create_weights_from_file(join_path(weights_dir, "conv3_1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv3_1_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto conv3_2 = convolution::create(
    {
        mem_format,
        {
            conv3_1,
            wo.create_weights_from_file(join_path(weights_dir, "conv3_2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv3_2_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto conv3_3 = convolution::create(
    {
        mem_format,
        {
            conv3_2,
            wo.create_weights_from_file(join_path(weights_dir, "conv3_3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv3_3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto pool3 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv3_3,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 2,2 },1 }, // kernel
        padding::zero
    });

    auto conv4_1 = convolution::create(
    {
        mem_format,
        {
            pool3,
            wo.create_weights_from_file(join_path(weights_dir, "conv4_1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv4_1_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto conv4_2 = convolution::create(
    {
        mem_format,
        {
            conv4_1,
            wo.create_weights_from_file(join_path(weights_dir, "conv4_2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv4_2_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto conv4_3 = convolution::create(
    {
        mem_format,
        {
            conv4_2,
            wo.create_weights_from_file(join_path(weights_dir, "conv4_3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv4_3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto pool4 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv4_3,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 2,2 },1 }, // kernel
        padding::zero
    });

    auto conv5_1 = convolution::create(
    {
        mem_format,
        {
            pool4,
            wo.create_weights_from_file(join_path(weights_dir, "conv5_1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv5_1_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1,1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto conv5_2 = convolution::create(
    {
        mem_format,
        {
            conv5_1,
            wo.create_weights_from_file(join_path(weights_dir, "conv5_2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv5_2_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1,1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto conv5_3 = convolution::create(
    {
        mem_format,
        {
            conv5_2,
            wo.create_weights_from_file(join_path(weights_dir, "conv5_3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv5_3_bias.nnd"),  file::bias),
        },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 1,1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto pool5 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv5_3,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 2,2 },1 }, // kernel
        padding::zero
    });

    auto fc6 = fully_connected::create(
    {
        fc_mem_format,
        pool5,
        wo.create_weights_from_file(join_path(weights_dir, "fc6_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc6_bias.nnd"),  file::bias),
        true,
        0
    });

    auto fc7 = fully_connected::create(
    {
        fc_mem_format,
        fc6,
        wo.create_weights_from_file(join_path(weights_dir, "fc7_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc7_bias.nnd"),  file::bias),
        true,
        0
    });

    auto fc8 = fully_connected::create(
    {
        fc_mem_format,
        fc7,
        wo.create_weights_from_file(join_path(weights_dir, "fc8_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc8_bias.nnd"),  file::bias),
        true,
        0
    });

    auto softmax = normalization::softmax::create(
    {
        output,
        fc8
    });

    return std::vector<std::pair<primitive, std::string>> {
        { reordered_input, "reorder"},
        { conv1_1, "conv1_1" },
        { conv1_2, "conv1_2" },
        { pool1, "pool1" },
        { conv2_1, "conv2_1" },
        { conv2_2, "conv2_2" },
        { pool2, "pool2" },
        { conv3_1, "conv3_1" },
        { conv3_2, "conv3_2" },
        { conv3_3, "conv3_3" },
        { pool3, "pool3" },
        { conv4_1, "conv4_1" },
        { conv4_2, "conv4_2" },
        { conv4_3, "conv4_3" },
        { pool4, "pool4" },
        { conv5_1, "conv5_1" },
        { conv5_2, "conv5_2" },
        { conv5_3, "conv5_3" },
        { pool5, "pool5" },
        { fc6, "fc6" },
        { fc7, "fc7" },
        { fc8, "fc8" },
        { softmax, "softmax" }
    };
}
