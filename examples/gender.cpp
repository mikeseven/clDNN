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

// Building age_gender network with loading weights & biases from file
// !!! commented layers will be used in the future !!!
std::vector<std::pair<primitive, std::string>> build_gender(const primitive& input, const primitive& output, const std::string& weights_dir, weights_optimizer& wo, bool use_half)
{
    auto mem_format = use_half ? memory::format::yxfb_f16 : memory::format::yxfb_f32;
    auto fc_mem_format = use_half ? memory::format::xb_f16 : memory::format::xb_f32;

    // create conversion to yxfb format and subtract mean values
    auto reordered_input = reorder::create(
    {
        mem_format,
        input.as<const memory&>().argument.size,
        input,
        { (float)104.0069879317889, (float)116.66876761696767, (float)122.6789143406786 },
        true
    });

    auto conv1 = convolution::create(
    {
        mem_format,
        {
            reordered_input,
            wo.create_weights_from_file(join_path(weights_dir, "conv1_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv1_biases.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto pool1 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv1,
        { 1,{ 1,1 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto conv2 = convolution::create(
    {
        mem_format,
        {
            pool1,
            wo.create_weights_from_file(join_path(weights_dir, "conv2_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv2_biases.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true, // negative slope for RELU
    });

    auto pool2 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv2,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    auto conv3 = convolution::create(
    {
        mem_format,
        {
            pool2,
            wo.create_weights_from_file(join_path(weights_dir, "conv3_weights.nnd"), file::convolution),
            wo.create_weights_from_file(join_path(weights_dir, "conv3_biases.nnd"),  file::bias),
        },
        { 0,{ 0, 0 }, 0 },
        { 1,{ 1, 1 }, 1 },
        padding::zero,
        1,
        true,
    });

    auto pool3 = pooling::create(
    {
        pooling::mode::max,
        mem_format,
        conv3,
        { 1,{ 2,2 },1 }, // strd
        { 1,{ 3,3 },1 }, // kernel
        padding::zero
    });

    /*
    auto fc1_a = fully_connected::create(
    {
        fc_mem_format,
        pool3,
        wo.create_weights_from_file(join_path(weights_dir, "fc1_a_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc1_a_biases.nnd"),  file::bias),
        true,
        0
    });
    */
    auto fc1_g = fully_connected::create(
    {
        fc_mem_format,
        pool3,
        wo.create_weights_from_file(join_path(weights_dir, "fc1_g_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc1_g_biases.nnd"),  file::bias),
        true,
        0
    });

  /*
    auto fc2_a = fully_connected::create(
    {
        fc_mem_format,
        fc1_a,
        wo.create_weights_from_file(join_path(weights_dir, "fc2_a_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc2_a_biases.nnd"),  file::bias),
        true,
        0
    });

    auto fc3_a = fully_connected::create(
    {
        fc_mem_format,
        fc2_a,
        wo.create_weights_from_file(join_path(weights_dir, "fc3_a_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc3_a_biases.nnd"),  file::bias),
        true,
        0
    });
    */
    auto fc3_g = fully_connected::create(
    {
        fc_mem_format,
        fc1_g,
        wo.create_weights_from_file(join_path(weights_dir, "fc3_g_weights.nnd"), file::fully_connected),
        wo.create_weights_from_file(join_path(weights_dir, "fc3_g_biases.nnd"),  file::bias),
        false,
        0
    });

    auto softmax = normalization::softmax::create(
    {
        output,
        fc3_g
    });

    return std::vector<std::pair<primitive, std::string>> {
        { reordered_input, "reorder"},
        { conv1, "conv1" },
        { pool1, "pool1" },
        { conv2, "conv2" },
        { pool2, "pool2" },
        { conv3, "conv3" },
        { pool3, "pool3" },
     // { fc1_a, "fc1_a" },
        { fc1_g, "fc1_g" },
     // { fc2_a, "fc2_a" },
     // { fc3_a, "fc3_a" },
        { fc3_g, "fc3_g" },
        { softmax, "softmax" }
   };
}
