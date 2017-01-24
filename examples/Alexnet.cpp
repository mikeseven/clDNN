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
#include "file.h"

#include <string>
#include <api/primitives/input_layout.hpp>
#include <api/primitives/reorder.hpp>
#include <api/primitives/convolution.hpp>
#include <api/primitives/pooling.hpp>
#include <api/primitives/normalization.hpp>
#include <api/primitives/fully_connected.hpp>
#include <api/primitives/softmax.hpp>

using namespace cldnn;

// Building AlexNet network with loading weights & biases from file
topology build_alexnet(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { format::byxf, { batch_size, 227, 227, 3 } };
    auto input = cldnn::input_layout("input", input_layout);

    // TODO: remove after enabling bfyx for all
    auto mem_format = format::yxfb;
    if (batch_size == 1 && input_layout.data_type == data_types::f32)
    {
        mem_format = format::bfyx;
    }

    // create conversion to yxfb format and subtract mean values
    tensor reorder_size = input_layout.size.transform(mem_format, 1);
    auto reorder_mean = file::create({ engine, join_path(weights_dir, "imagenet_mean.nnd"), file::mean });
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, reorder_size },
        reorder_mean);

    auto conv1_weights = file::create({ engine, join_path(weights_dir, "conv1_weights.nnd"), file::convolution });
    auto conv1_biases = file::create({ engine, join_path(weights_dir, "conv1_biases.nnd"), file::bias });
    auto conv1 = convolution(
        "conv1",
        reordered_input,
        { conv1_weights },
        { conv1_biases },
        { format::yx, {0,0} },
        { format::yx, {4,4} },
        true);

    auto pool1 = pooling(
        "pool1",
        conv1,
        pooling_mode::max,
        { format::yx, {2,2} }, // strd
        { format::yx, {3,3} }); // kernel

    auto lrn1 = normalization(
        "lrn1",
        pool1,
        5,
        1.0f,
        0.00002f,
        0.75f);

    auto conv2_g1_weights = file::create({ engine, join_path(weights_dir, "conv2_g1_weights.nnd"), file::convolution });
    auto conv2_g1_biases = file::create({ engine, join_path(weights_dir, "conv2_g1_biases.nnd"), file::bias });
    auto conv2_g2_weights = file::create({ engine, join_path(weights_dir, "conv2_g2_weights.nnd"), file::convolution });
    auto conv2_g2_biases = file::create({ engine, join_path(weights_dir, "conv2_g2_biases.nnd"), file::bias });
    auto conv2_group2 = convolution(
        "conv2_group2",
        lrn1,
        { conv2_g1_weights, conv2_g2_weights },
        { conv2_g1_biases, conv2_g2_biases },
        { format::yx, {-2, -2} },
        { format::yx, {1, 1} },
        true);

    auto pool2 = pooling(
        "pool2",
        conv2_group2,
        pooling_mode::max,
        { format::yx, { 2,2 } }, // strd
        { format::yx, { 3,3 } }); // kernel

    auto lrn2 = normalization(
        "lrn2",
        pool2,
        5,
        1.0f,
        0.00002f,
        0.75f);

    auto conv3_weights = file::create({ engine, join_path(weights_dir, "conv3_weights.nnd"), file::convolution });
    auto conv3_biases = file::create({ engine, join_path(weights_dir, "conv3_biases.nnd"), file::bias });
    auto conv3 = convolution(
        "conv3",
        lrn2,
        { conv3_weights },
        { conv3_biases },
        { format::yx, {-1, -1} },
        { format::yx, {1,1} },
        true);

    auto conv4_g1_weights = file::create({ engine, join_path(weights_dir, "conv4_g1_weights.nnd"), file::convolution });
    auto conv4_g1_biases = file::create({ engine, join_path(weights_dir, "conv4_g1_biases.nnd"), file::bias });
    auto conv4_g2_weights = file::create({ engine, join_path(weights_dir, "conv4_g2_weights.nnd"), file::convolution });
    auto conv4_g2_biases = file::create({ engine, join_path(weights_dir, "conv4_g2_biases.nnd"), file::bias });
    auto conv4_group2 = convolution(
        "conv4_group2",
        conv3,
        { conv4_g1_weights, conv4_g2_weights },
        { conv4_g1_biases, conv4_g2_biases },
        { format::yx, {-1,-1} },
        { format::yx, {1,1} },
        true);

    auto conv5_g1_weights = file::create({ engine, join_path(weights_dir, "conv5_g1_weights.nnd"), file::convolution });
    auto conv5_g1_biases = file::create({ engine, join_path(weights_dir, "conv5_g1_biases.nnd"), file::bias });
    auto conv5_g2_weights = file::create({ engine, join_path(weights_dir, "conv5_g2_weights.nnd"), file::convolution });
    auto conv5_g2_biases = file::create({ engine, join_path(weights_dir, "conv5_g2_biases.nnd"), file::bias });
    auto conv5_group2 = convolution(
        "conv5_group2",
        conv4_group2,
        { conv5_g1_weights, conv5_g2_weights },
        { conv5_g1_biases, conv5_g2_biases },
        { format::yx,{-1,-1} },
        { format::yx,{1,1} },
        true);

    auto pool5 = pooling(
        "pool5",
        conv5_group2,
        pooling_mode::max,
        { format::xy,{ 2, 2 } }, // strd
        { format::xy,{ 3, 3 } }); // kernel

    auto fc6_weights = file::create({ engine, join_path(weights_dir, "fc6_weights.nnd"), file::fully_connected });
    auto fc6_biases = file::create({ engine, join_path(weights_dir, "fc6_biases.nnd"), file::bias });
    auto fc6 = fully_connected(
        "fc6",
        pool5,
        fc6_weights,
        fc6_biases,
        true);

    auto fc7_weights = file::create({ engine, join_path(weights_dir, "fc7_weights.nnd"), file::fully_connected });
    auto fc7_biases = file::create({ engine, join_path(weights_dir, "fc7_biases.nnd"), file::bias });
    auto fc7 = fully_connected(
        "fc7",
        fc6,
        fc7_weights,
        fc7_biases,
        true);

    auto fc8_weights = file::create({ engine, join_path(weights_dir, "fc8_weights.nnd"), file::fully_connected });
    auto fc8_biases = file::create({ engine, join_path(weights_dir, "fc8_biases.nnd"), file::bias });
    auto fc8 = fully_connected(
        "fc8",
        fc7,
        fc8_weights,
        fc8_biases,
        true);

    auto softmax = cldnn::softmax(
        "output",
        fc8);

    return topology(
        input,
        reordered_input,
        conv1,
        pool1,
        lrn1,
        conv2_group2,
        pool2,
        lrn2,
        conv3,
        conv4_group2,
        conv5_group2,
        pool5,
        fc6,
        fc7,
        fc8,
        softmax
    );
}