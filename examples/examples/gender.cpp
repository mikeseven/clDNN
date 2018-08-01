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

#include "common_tools.h"
#include "file.h"

#include <string>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/reorder.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/softmax.hpp>

using namespace cldnn;

// Building age_gender network with loading weights & biases from file
// !!! commented layers will be used in the future !!!
cldnn::topology build_gender(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    input_layout.size = { batch_size, 3, 86, 86 };
    auto input = cldnn::input_layout("input", input_layout);

    // subtract mean values
    auto reordered_input = reorder(
        "reorder",
        input,
        layout( input_layout.data_type, input_layout.format, input_layout.size),
        std::vector<float>{ (float)104.0069879317889, (float)116.66876761696767, (float)122.6789143406786 });

    auto conv1_weights = file::create({ engine, join_path(weights_dir, "conv1_weights.nnd")});
    auto conv1_bias = file::create({ engine, join_path(weights_dir, "conv1_bias.nnd")});
    auto conv1 = convolution(
        "conv1",
        reordered_input,
        { conv1_weights },
        { conv1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto pool1 = pooling(
        "pool1",
        conv1,
        pooling_mode::max,
        { 1,1,3,3 },  // kernel
        { 1,1,1,1 }); // strd

    auto conv2_weights = file::create({ engine, join_path(weights_dir, "conv2_weights.nnd")});
    auto conv2_bias = file::create({ engine, join_path(weights_dir, "conv2_bias.nnd")});
    auto conv2 = convolution(
        "conv2",
        pool1,
        { conv2_weights },
        { conv2_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto pool2 = pooling(
        "pool2",
        conv2,
        pooling_mode::max,
        { 1,1,3,3 },  // kernel
        { 1,1,2,2 }); // strd

    auto conv3_weights = file::create({ engine, join_path(weights_dir, "conv3_weights.nnd")});
    auto conv3_bias = file::create({ engine, join_path(weights_dir, "conv3_bias.nnd")});
    auto conv3 = convolution(
        "conv3",
        pool2,
        { conv3_weights },
        { conv3_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto pool3 = pooling(
        "pool3",
        conv3,
        pooling_mode::max,
        { 1,1,3,3 },  // kernel
        { 1,1,2,2 }); // strd

    auto fc1_g_weights = file::create({ engine, join_path(weights_dir, "fc1_g_weights.nnd")});
    auto fc1_g_bias = file::create({ engine, join_path(weights_dir, "fc1_g_bias.nnd")});
    auto fc1_g = fully_connected(
        "fc1_g",
        pool3,
        fc1_g_weights,
        fc1_g_bias,
        true,
        0);

    auto fc3_g_weights = file::create({ engine, join_path(weights_dir, "fc3_g_weights.nnd")});
    auto fc3_g_bias = file::create({ engine, join_path(weights_dir, "fc3_g_bias.nnd")});
    auto fc3_g = fully_connected(
        "fc3_g",
        fc1_g,
        fc3_g_weights,
        fc3_g_bias,
        false,
        0);

    auto softmax = cldnn::softmax(
        "output",
        fc3_g);

    return topology{
        input,
        reordered_input,
        conv1, conv1_weights, conv1_bias,
        pool1,
        conv2, conv2_weights, conv2_bias,
        pool2,
        conv3, conv3_weights, conv3_bias,
        pool3,
        fc1_g, fc1_g_weights, fc1_g_bias,
        fc3_g, fc3_g_weights, fc3_g_bias,
        softmax
   };
}