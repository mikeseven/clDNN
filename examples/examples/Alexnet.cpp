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
#include <api/CPP/lrn.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/softmax.hpp>

using namespace cldnn;

// Building AlexNet network with loading weights & biases from file
topology build_alexnet(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { batch_size, 3, 227, 227 };
    auto input = cldnn::input_layout("input", input_layout);

    // subtract mean values
    auto reorder_mean = file::create({ engine, join_path(weights_dir, "imagenet_mean.nnd")});
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, input_layout.format, input_layout.size },
        reorder_mean);

    auto conv1_weights = file::create({ engine, join_path(weights_dir, "conv1_weights.nnd")});
    auto conv1_biases = file::create({ engine, join_path(weights_dir, "conv1_biases.nnd")});
    auto conv1 = convolution(
        "conv1",
        reordered_input,
        { conv1_weights },
        { conv1_biases },
        { 1,1,4,4 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto pool1 = pooling(
        "pool1",
        conv1,
        pooling_mode::max,
        { 1,1,3,3 }, // kernel
        { 1,1,2,2 }); // strd
        
    auto lrn1 = lrn(
        "lrn1",
        pool1,
        5,
        1.0f,
        0.0001f,
        0.75f,
        cldnn_lrn_norm_region_across_channel);

    auto conv2_g1_weights = file::create({ engine, join_path(weights_dir, "conv2_g1_weights.nnd")});
    auto conv2_g1_biases = file::create({ engine, join_path(weights_dir, "conv2_g1_biases.nnd")});
    auto conv2_g2_weights = file::create({ engine, join_path(weights_dir, "conv2_g2_weights.nnd")});
    auto conv2_g2_biases = file::create({ engine, join_path(weights_dir, "conv2_g2_biases.nnd")});
    auto conv2_group2 = convolution(
        "conv2_group2",
        lrn1,
        { conv2_g1_weights, conv2_g2_weights },
        { conv2_g1_biases, conv2_g2_biases },
        { 1,1,1,1 },
        { 0,0,-2,-2 },
        { 1,1,1,1 },
        true);

    auto pool2 = pooling(
        "pool2",
        conv2_group2,
        pooling_mode::max,
        { 1,1,3,3 }, // kernel
        { 1,1,2,2 }); // strd

    auto lrn2 = lrn(
        "lrn2",
        pool2,
        5,
        1.0f,
        0.0001f,
        0.75f,
        cldnn_lrn_norm_region_across_channel);

    auto conv3_weights = file::create({ engine, join_path(weights_dir, "conv3_weights.nnd")});
    auto conv3_biases = file::create({ engine, join_path(weights_dir, "conv3_biases.nnd")});
    auto conv3 = convolution(
        "conv3",
        lrn2,
        { conv3_weights },
        { conv3_biases },
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv4_g1_weights = file::create({ engine, join_path(weights_dir, "conv4_g1_weights.nnd")});
    auto conv4_g1_biases = file::create({ engine, join_path(weights_dir, "conv4_g1_biases.nnd")});
    auto conv4_g2_weights = file::create({ engine, join_path(weights_dir, "conv4_g2_weights.nnd")});
    auto conv4_g2_biases = file::create({ engine, join_path(weights_dir, "conv4_g2_biases.nnd")});
    auto conv4_group2 = convolution(
        "conv4_group2",
        conv3,
        { conv4_g1_weights, conv4_g2_weights },
        { conv4_g1_biases, conv4_g2_biases },
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv5_g1_weights = file::create({ engine, join_path(weights_dir, "conv5_g1_weights.nnd")});
    auto conv5_g1_biases = file::create({ engine, join_path(weights_dir, "conv5_g1_biases.nnd")});
    auto conv5_g2_weights = file::create({ engine, join_path(weights_dir, "conv5_g2_weights.nnd")});
    auto conv5_g2_biases = file::create({ engine, join_path(weights_dir, "conv5_g2_biases.nnd")});
    auto conv5_group2 = convolution(
        "conv5_group2",
        conv4_group2,
        { conv5_g1_weights, conv5_g2_weights },
        { conv5_g1_biases, conv5_g2_biases },
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto pool5 = pooling(
        "pool5",
        conv5_group2,
        pooling_mode::max,
        { 1,1,3,3 }, // kernel
        { 1,1,2,2 }); // strd

    auto fc6_weights = file::create({ engine, join_path(weights_dir, "fc6_weights.nnd")});
    auto fc6_biases = file::create({ engine, join_path(weights_dir, "fc6_biases.nnd")});
    auto fc6 = fully_connected(
        "fc6",
        pool5,
        fc6_weights,
        fc6_biases,
        true);

    auto fc7_weights = file::create({ engine, join_path(weights_dir, "fc7_weights.nnd")});
    auto fc7_biases = file::create({ engine, join_path(weights_dir, "fc7_biases.nnd")});
    auto fc7 = fully_connected(
        "fc7",
        fc6,
        fc7_weights,
        fc7_biases,
        true);

    auto fc8_weights = file::create({ engine, join_path(weights_dir, "fc8_weights.nnd")});
    auto fc8_biases = file::create({ engine, join_path(weights_dir, "fc8_biases.nnd")});
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
        reordered_input, reorder_mean,
        conv1, conv1_weights, conv1_biases,
        pool1,
        lrn1,
        conv2_group2, conv2_g1_weights, conv2_g1_biases, conv2_g2_weights, conv2_g2_biases,
        pool2,
        lrn2,
        conv3, conv3_weights, conv3_biases,
        conv4_group2, conv4_g1_weights, conv4_g1_biases, conv4_g2_weights, conv4_g2_biases,
        conv5_group2, conv5_g1_weights, conv5_g1_biases, conv5_g2_weights, conv5_g2_biases,
        pool5,
        fc6, fc6_weights, fc6_biases,
        fc7, fc7_weights, fc7_biases,
        fc8, fc8_weights, fc8_biases,
        softmax
    );
}
