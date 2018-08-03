// Copyright (c) 2018 Intel Corporation
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


#include <api/CPP/input_layout.hpp>
#include <api/CPP/reorder.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/softmax.hpp>
#include <api/CPP/eltwise.hpp>

#include <string>

#include "common_tools.h"
#include "file.h"


using namespace cldnn;
using namespace std;


static primitive_id add_conv_layer(const string& weights_dir, const engine& engine, topology& topology_inst,
                                   const string& layer_name, const primitive_id& input,
                                   const tensor& padding = {0, 0, 0, 0}, const tensor& stride = {1, 1, 1, 1},
                                   const bool add_relu = true)
{
    auto weights_data = file::create({engine, join_path(weights_dir, layer_name + "_weights.nnd")});
    auto bias_data    = file::create({engine, join_path(weights_dir, layer_name + "_bias.nnd")});

    auto conv_layer = convolution(
        layer_name,
        input,
        {weights_data},
        {bias_data},
        stride,
        padding,
        {1, 1, 1, 1},
        add_relu);

    topology_inst.add(weights_data, bias_data, conv_layer);

    return conv_layer;
}

static primitive_id add_conv_layer_no_relu(const string& weights_dir, const engine& engine, topology& topology_inst,
                                           const string& layer_name, const primitive_id& input,
                                           const tensor& padding = {0, 0, 0, 0}, const tensor& stride = {1, 1, 1, 1})
{
    return add_conv_layer(weights_dir, engine, topology_inst, layer_name, input, padding, stride, false);
}

static primitive_id add_residual_layers(const string& weights_dir, const engine& engine, topology& topology_inst,
                                        const string& res_name, const primitive_id& input,
                                        bool conv_in_branch1, const tensor& res_stride = {1, 1, 1, 1})
{
    auto conv_branch2a = add_conv_layer(weights_dir, engine, topology_inst, res_name + "_branch2a",
                                        input, {0, 0, 0, 0}, res_stride);
    auto conv_branch2b = add_conv_layer(weights_dir, engine, topology_inst, res_name + "_branch2b",
                                        conv_branch2a, {0, 0, -1, -1});
    auto conv_branch2c = add_conv_layer_no_relu(weights_dir, engine, topology_inst, res_name + "_branch2c",
                                                conv_branch2b);

    primitive_id conv_branch1 = input;
    if (conv_in_branch1)
        conv_branch1 = add_conv_layer_no_relu(weights_dir, engine, topology_inst, res_name + "_branch1",
                                              input, {0, 0, 0, 0}, res_stride);

    auto res_sum = eltwise(res_name, conv_branch1, conv_branch2c, eltwise_mode::sum, true);
    topology_inst.add(res_sum);

    return res_sum;
}


cldnn::topology build_resnet50(const std::string& weights_dir, const cldnn::engine& engine,
                               cldnn::layout& input_layout, int32_t batch_size, const bool mean_subtract)
{
    // [224x224x3xF] input->convolution(conv1)->relu(conv1_relu)->pooling[max](pool1) [56x56x64xF].
    input_layout.size = {batch_size, 3, 224, 224};
    auto input        = cldnn::input_layout("input", input_layout);
    topology topology_inst{input};

    primitive_id corrected_input = input;
    if (mean_subtract)
    {
        // Subtract mean values if necessary.
        auto reorder_mean = file::create({engine, join_path(weights_dir, "resnet50_mean.nnd")});
        auto reordered_input = reorder(
            "reorder",
            input,
            { input_layout.data_type, format::bfyx, input_layout.size },
            reorder_mean);
        topology_inst.add(reorder_mean, reordered_input);
        corrected_input = reordered_input;
    }

    auto conv1 = add_conv_layer(weights_dir, engine, topology_inst, "conv1", corrected_input,
                                {0, 0, -3, -3}, {1, 1, 2, 2});

    auto pool1 = pooling(
        "pool1",
        conv1,
        pooling_mode::max,
        {1, 1, 3, 3},
        {1, 1, 2, 2});
    topology_inst.add(pool1);

    // [56x56x64xF] res2a->res2b->res2c [56x56x256xF].
    auto res2a = add_residual_layers(weights_dir, engine, topology_inst, "res2a", pool1, true);
    auto res2b = add_residual_layers(weights_dir, engine, topology_inst, "res2b", res2a, false);
    auto res2c = add_residual_layers(weights_dir, engine, topology_inst, "res2c", res2b, false);

    // [56x56x256xF] res3a->res3b->res3c->res3d [28x28x512xF].
    auto res3a = add_residual_layers(weights_dir, engine, topology_inst, "res3a", res2c, true, {1, 1, 2, 2});
    auto res3b = add_residual_layers(weights_dir, engine, topology_inst, "res3b", res3a, false);
    auto res3c = add_residual_layers(weights_dir, engine, topology_inst, "res3c", res3b, false);
    auto res3d = add_residual_layers(weights_dir, engine, topology_inst, "res3d", res3c, false);

    // [28x28x512xF] res4a->res4b->res4c->res4d->res4e->res4f [14x14x1024xF].
    auto res4a = add_residual_layers(weights_dir, engine, topology_inst, "res4a", res3d, true, {1, 1, 2, 2});
    auto res4b = add_residual_layers(weights_dir, engine, topology_inst, "res4b", res4a, false);
    auto res4c = add_residual_layers(weights_dir, engine, topology_inst, "res4c", res4b, false);
    auto res4d = add_residual_layers(weights_dir, engine, topology_inst, "res4d", res4c, false);
    auto res4e = add_residual_layers(weights_dir, engine, topology_inst, "res4e", res4d, false);
    auto res4f = add_residual_layers(weights_dir, engine, topology_inst, "res4f", res4e, false);

    // [14x14x1024xF] res5a->res5b->res5c [7x7x2048xF].
    auto res5a = add_residual_layers(weights_dir, engine, topology_inst, "res5a", res4f, true, {1, 1, 2, 2});
    auto res5b = add_residual_layers(weights_dir, engine, topology_inst, "res5b", res5a, false);
    auto res5c = add_residual_layers(weights_dir, engine, topology_inst, "res5c", res5b, false);

    // TODO: Avergae no padding?
    auto pool5 = pooling(
        "pool5",
        res5c,
        pooling_mode::average,
        {1, 1, 7, 7},
        {1, 1, 1, 1});

    auto fc1000_weights_data = file::create({engine, join_path(weights_dir, "fc1000_weights.nnd")});
    auto fc1000_bias_data    = file::create({engine, join_path(weights_dir, "fc1000_bias.nnd")});
    auto fc1000 = fully_connected(
        "fc1000",
        pool5,
        fc1000_weights_data,
        fc1000_bias_data);

    auto softmax = cldnn::softmax(
        "output",
        fc1000);

    topology_inst.add(pool5,
                      fc1000_weights_data, fc1000_bias_data, fc1000,
                      softmax);
    return topology_inst;
}
