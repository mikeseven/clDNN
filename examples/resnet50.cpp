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
#include <api/CPP/activation.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/reorder.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/lrn.hpp>
#include <api/CPP/batch_norm.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/concatenation.hpp>
#include <api/CPP/softmax.hpp>
#include <api/CPP/scale.hpp>
#include <api/CPP/eltwise.hpp>

using namespace cldnn;
using namespace std;

void add_conv(const engine& engine, topology& tpl, const string& name, const primitive_id& input,
    const std::string& weights_dir, const tensor& padding = { 0,0,0,0 }, const tensor& stride = { 1,1,1,1 })
{

    auto w = file::create({ engine, join_path(weights_dir, name + "_weights.nnd") });
    auto b = file::create({ engine, join_path(weights_dir, name + "_bias.nnd") });
    auto c = convolution(
        name,
        input,
        { w },
        { b },
        stride,
        padding,
        { 1, 1, 1, 1 },
        true);
    tpl.add(w, b, c);
}

void add_residual(const engine& engine, topology& tpl,const string& branch, const primitive_id& input, bool conv_in_branch1, const std::string& weights_dir, tensor stride2a = { 1,1,1,1 })
{
    add_conv(engine, tpl, branch + "_branch2a", input, weights_dir);
    add_conv(engine, tpl, branch + "_branch2b", branch + "_branch2a", weights_dir, { 0,0,-1,-1 });
    add_conv(engine, tpl, branch + "_branch2c", branch + "_branch2b", weights_dir);

    primitive_id branch1 = input;

    if (conv_in_branch1)
    {
        add_conv(engine, tpl, branch + "_branch1", input, weights_dir);
        branch1 = branch + "_branch1";
    }
    const primitive_id branch2 = branch + "_branch2c";
    tpl.add(eltwise(branch, { branch1, branch2 }, eltwise_mode::sum, true));
}

cldnn::topology build_resnet50(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { batch_size, 3, 224, 224 };
    auto input = cldnn::input_layout("input", input_layout);
    cldnn::topology topology{
        input };
   
    // subtract mean values
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, format::bfyx, input_layout.size });

    auto conv1_w = file::create({ engine, join_path(weights_dir, "conv1_weights.nnd")});
    auto conv1_b = file::create({ engine, join_path(weights_dir, "conv1_bias.nnd")});
    auto conv1 = convolution(
        "conv1",
        "reorder",
        { conv1_w },
        { conv1_b },
        { 1, 1, 2, 2 },
        { 0, 0, -3, -3 },
        { 1, 1, 1, 1 },
        true);

    auto pool1 = pooling(
        "pool1",
        "conv1",
        pooling_mode::max,
        { 1,1,3,3 },  // kernel
        { 1,1,2,2 }); // strd

    topology.add(reordered_input,
        conv1_w, conv1_b, conv1, pool1);

    add_residual(engine, topology, "res2a", "pool1", true, weights_dir);

    auto output = activation("output", "reorder", activation_relu);
    topology.add(output);
    return topology;
}
