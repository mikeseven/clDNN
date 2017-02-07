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
#include <api/primitives/input_layout.hpp>
#include <api/primitives/reorder.hpp>
#include <api/primitives/convolution.hpp>
#include <api/primitives/pooling.hpp>
#include <api/primitives/fully_connected.hpp>
#include <api/primitives/softmax.hpp>

using namespace cldnn;

// Building age_gender network with loading weights & biases from file
// !!! commented layers will be used in the future !!!
cldnn::topology build_gender(const std::string& weights_dir, weights_optimizer& wo, cldnn::layout& input_layout, int32_t batch_size, bool use_bfyx)
{
    input_layout.size = { format::byxf,{ batch_size, 86, 86, 3 } };
    auto input = cldnn::input_layout("input", input_layout);

    auto mem_format = use_bfyx ? format::bfyx : format::yxfb;

    // create conversion to yxfb format and subtract mean values
    tensor reorder_size = input_layout.size.transform(mem_format, 1);
    auto reordered_input = reorder(
        "reorder",
        input,
        layout(input_layout.data_type, reorder_size),
        std::vector<float>{ (float)104.0069879317889, (float)116.66876761696767, (float)122.6789143406786 });

    auto conv1_weights = wo.create_weights_from_file(join_path(weights_dir, "conv1_weights.nnd"));
    auto conv1_bias = wo.create_weights_from_file(join_path(weights_dir, "conv1_bias.nnd"));
    auto conv1 = convolution(
        "conv1",
        reordered_input,
        { conv1_weights },
        { conv1_bias },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto pool1 = pooling(
        "pool1",
        conv1,
        pooling_mode::max,
        { format::yx,{ 1,1 } },  // strd
        { format::yx,{ 3,3 } }); // kernel

    auto conv2_weights = wo.create_weights_from_file(join_path(weights_dir, "conv2_weights.nnd"));
    auto conv2_bias = wo.create_weights_from_file(join_path(weights_dir, "conv2_bias.nnd"));
    auto conv2 = convolution(
        "conv2",
        pool1,
        { conv2_weights },
        { conv2_bias },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto pool2 = pooling(
        "pool2",
        conv2,
        pooling_mode::max,
        { format::yx,{ 2,2 } },  // strd
        { format::yx,{ 3,3 } }); // kernel

    auto conv3_weights = wo.create_weights_from_file(join_path(weights_dir, "conv3_weights.nnd"));
    auto conv3_bias = wo.create_weights_from_file(join_path(weights_dir, "conv3_bias.nnd"));
    auto conv3 = convolution(
        "conv3",
        pool2,
        { conv3_weights },
        { conv3_bias },
        { format::yx,{ 0,0 } },
        { format::yx,{ 1,1 } },
        true);

    auto pool3 = pooling(
        "pool3",
        conv3,
        pooling_mode::max,
        { format::yx,{ 2,2 } },  // strd
        { format::yx,{ 3,3 } }); // kernel

    auto fc1_g_weights = wo.create_weights_from_file(join_path(weights_dir, "fc1_g_weights.nnd"));
    auto fc1_g_bias = wo.create_weights_from_file(join_path(weights_dir, "fc1_g_bias.nnd"));
    auto fc1_g = fully_connected(
        "fc1_g",
        pool3,
        fc1_g_weights,
        fc1_g_bias,
        true,
        0);

    auto fc3_g_weights = wo.create_weights_from_file(join_path(weights_dir, "fc3_g_weights.nnd"));
    auto fc3_g_bias = wo.create_weights_from_file(join_path(weights_dir, "fc3_g_bias.nnd"));
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
        conv1,
        pool1,
        conv2,
        pool2,
        conv3,
        pool3,
        fc1_g,
        fc3_g,
        softmax
    };
}
