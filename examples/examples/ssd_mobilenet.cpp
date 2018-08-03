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
#include <api/CPP/scale.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/concatenation.hpp>
#include <api/CPP/softmax.hpp>
#include <api/CPP/prior_box.hpp>
#include <api/CPP/permute.hpp>
#include <api/CPP/reshape.hpp>
#include <api/CPP/detection_output.hpp>

#include "common_tools.h"
#include "file.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>


using namespace cldnn;
using namespace std;


static primitive_id add_conv_layer(const string& weights_dir, const engine& engine, topology& topology_inst,
                                   const string& layer_name, const string& weights_root, const primitive_id& input,
                                   const tensor& padding = {0, 0, 0, 0}, const tensor& stride = {1, 1, 1, 1},
                                   const size_t split = 1, const bool add_relu = true)
{
    vector<primitive_id> weights_data_groups;
    vector<primitive_id> bias_data_groups;

    if (split <= 1)
    {
        auto weights_data = file::create({engine, join_path(weights_dir, weights_root + "_weights.nnd")});
        auto bias_data    = file::create({engine, join_path(weights_dir, weights_root + "_bias.nnd")});

        weights_data_groups.push_back(weights_data);
        bias_data_groups.push_back(bias_data);

        topology_inst.add(weights_data, bias_data);
    }
    else
    {
        for (size_t gi = 1; gi <= split; ++gi)
        {
            auto weights_data = file::create({
                engine,
                join_path(weights_dir, weights_root + "_g" + std::to_string(gi) + "_weights.nnd")
            });
            auto bias_data    = file::create({
                engine,
                join_path(weights_dir, weights_root + "_g" + std::to_string(gi) + "_bias.nnd")
            });

            weights_data_groups.push_back(weights_data);
            bias_data_groups.push_back(bias_data);

            topology_inst.add(weights_data, bias_data);
        }
    }

    auto conv_layer = convolution(
        layer_name,
        input,
        weights_data_groups,
        bias_data_groups,
        stride,
        padding,
        {1, 1, 1, 1},
        add_relu
    );
    topology_inst.add(conv_layer);

    return conv_layer;
}


static primitive_id add_mul_layer(const engine& engine, topology& topology_inst,
                                  const string& layer_name, const primitive_id& input, const float scale = 1.0f)
{
    auto scale_mem = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, 1, 1}});
    {
        auto scale_ptr = scale_mem.pointer<float>();
        scale_ptr[0] = scale;
    }
    auto scale_data = cldnn::data(layer_name + "_scale_data", scale_mem);

    auto mul_layer = cldnn::scale(
        layer_name,
        input,
        scale_data
    );
    topology_inst.add(scale_data, mul_layer);

    return mul_layer;
}


// --------------------------------------------------------------------------------------------------------------------
// Mobilenet-specific:

static primitive_id add_dw_pw_conv_pair(const string& weights_dir, const engine& engine, topology& topology_inst,
                                        const string& root_name, const primitive_id& input,
                                        const tensor& dw_stride = {1, 1, 1, 1}, const size_t dw_split = 1,
                                        const tensor& dw_padding = {0, 0, -1, -1})
{
    auto conv_dw = add_conv_layer(weights_dir, engine, topology_inst, root_name + "_dw", root_name + "%2Fdw", input,
                                  dw_padding, dw_stride, dw_split);
    auto conv_pw = add_conv_layer(weights_dir, engine, topology_inst, root_name, root_name, conv_dw);
    return conv_pw;
}

static primitive_id add_reduce_pair(const string& weights_dir, const engine& engine, topology& topology_inst,
                                    const string& root_name, const primitive_id& input,
                                    const tensor& reduce_stride = {1, 1, 2, 2},
                                    const tensor& reduce_padding = {0, 0, -1, -1})
{
    auto conv_1 = add_conv_layer(weights_dir, engine, topology_inst, root_name + "_1", root_name + "_1", input);
    auto conv_2 = add_conv_layer(weights_dir, engine, topology_inst, root_name + "_2", root_name + "_2", conv_1,
                                 reduce_padding, reduce_stride);
    return conv_2;
}

static primitive_id add_mbox_processor(const string& weights_dir, const engine& engine, int32_t batch_size,
                                       topology& topology_inst, const string& root_name,
                                       const vector<pair<primitive_id, size_t>>& input_and_size_pairs,
                                       const vector<uint16_t>& input_order = {0, 2, 3, 1})
{
    vector<primitive_id> concat_inputs;
    for (const auto& input_and_size_pair : input_and_size_pairs)
    {
        auto input_root = input_and_size_pair.first + "_" + root_name;

        auto input_conv = add_conv_layer(weights_dir, engine, topology_inst, input_root, input_root,
                                         input_and_size_pair.first,
                                         {0, 0, 0, 0}, {1, 1, 1, 1}, 1, false);
        auto input_perm = permute(
            input_root + "_perm",
            input_conv,
            input_order
        );
        auto input_flat = reshape(
            input_root + "_flat",
            input_perm,
            {batch_size, static_cast<tensor::value_type>(input_and_size_pair.second), 1, 1}
        );

        concat_inputs.push_back(input_flat);

        topology_inst.add(input_perm, input_flat);
    }

    auto mbox_proc = concatenation(
        root_name,
        concat_inputs,
        concatenation::along_f
    );
    topology_inst.add(mbox_proc);

    return mbox_proc;
}

namespace
{
struct input_priorbox_settings
{
    primitive_id  input;

    vector<float> min_sizes;
    vector<float> max_sizes;
    vector<float> aspect_ratios;
};
}

static primitive_id add_mbox_priorbox(const layout& input_layout, topology& topology_inst, const string& root_name,
                                      const vector<input_priorbox_settings>& input_prior_infos,
                                      const vector<float>& input_variance = {0.1f, 0.1f, 0.2f, 0.2f},
                                      const bool input_flip = true, const bool input_clip = false)
{
    vector<primitive_id> concat_inputs;
    for (const auto& input_prior_info : input_prior_infos)
    {
        auto input_root = input_prior_info.input + "_" + root_name;

        auto input_priorbox = prior_box(
            input_root,
            input_prior_info.input,
            input_layout.size,
            input_prior_info.min_sizes,
            input_prior_info.max_sizes,
            input_prior_info.aspect_ratios,
            input_flip,
            input_clip,
            input_variance
        );

        concat_inputs.push_back(input_priorbox);

        topology_inst.add(input_priorbox);
    }

    auto mbox_priorbox = concatenation(
        root_name,
        concat_inputs,
        concatenation::along_y
    );
    topology_inst.add(mbox_priorbox);

    return mbox_priorbox;
}


// Building SSD MobileNet network with loading weights & biases from file
cldnn::topology build_ssd_mobilenet(const std::string& weights_dir, const cldnn::engine& engine,
                                    cldnn::layout& input_layout, int32_t batch_size)
{
    // [300x300x3xB]
    input_layout.size = {batch_size, 3, 300, 300};
    auto input        = cldnn::input_layout("input", input_layout);
    cldnn::topology topology_inst{input};

    auto mul1_340 = add_mul_layer(engine, topology_inst, "mul1_340", input, 0.016999864f);

    // Initial feature extractor.
    auto conv0 = add_conv_layer(weights_dir, engine, topology_inst, "conv0", "conv0", mul1_340,
                                {0, 0, -1, -1}, {1, 1, 2, 2});

    // Depthwise-pointwise feature-extraction chain (convolutions).
    auto conv1  = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv1",  conv0,  {1, 1, 1, 1}, 32);

    auto conv2  = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv2",  conv1,  {1, 1, 2, 2}, 64);

    auto conv3  = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv3",  conv2,  {1, 1, 1, 1}, 128);
    auto conv4  = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv4",  conv3,  {1, 1, 2, 2}, 128);

    auto conv5  = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv5",  conv4,  {1, 1, 1, 1}, 256);
    auto conv6  = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv6",  conv5,  {1, 1, 2, 2}, 256);

    auto conv7  = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv7",  conv6,  {1, 1, 1, 1}, 512);
    auto conv8  = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv8",  conv7,  {1, 1, 1, 1}, 512);
    auto conv9  = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv9",  conv8,  {1, 1, 1, 1}, 512);
    auto conv10 = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv10", conv9,  {1, 1, 1, 1}, 512);
    auto conv11 = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv11", conv10, {1, 1, 1, 1}, 512);
    auto conv12 = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv12", conv11, {1, 1, 2, 2}, 512);

    auto conv13 = add_dw_pw_conv_pair(weights_dir, engine, topology_inst, "conv13", conv12, {1, 1, 1, 1}, 1024);

    // Two-stage reductors chain (convolutions) for multi-box localizers/classifiers/prioritetizers.
    auto conv14 = add_reduce_pair(weights_dir, engine, topology_inst, "conv14", conv13);
    auto conv15 = add_reduce_pair(weights_dir, engine, topology_inst, "conv15", conv14);
    auto conv16 = add_reduce_pair(weights_dir, engine, topology_inst, "conv16", conv15);
    auto conv17 = add_reduce_pair(weights_dir, engine, topology_inst, "conv17", conv16);

    // Multi-box locator.
    auto mbox_loc = add_mbox_processor(weights_dir, engine, batch_size, topology_inst, "mbox_loc", {
        {conv11, 4332},
        {conv13, 2400},
        {conv14, 600},
        {conv15, 216},
        {conv16, 96},
        {conv17, 24}
    });

    // Multi-box classifier.
    auto mbox_conf = add_mbox_processor(weights_dir, engine, batch_size, topology_inst, "mbox_conf", {
        {conv11, 22743},
        {conv13, 12600},
        {conv14, 3150},
        {conv15, 1134},
        {conv16, 504},
        {conv17, 126}
    });

    // Multi-box priorbox (prioritetizer / ROI selectors).
    auto mbox_priorbox = add_mbox_priorbox(input_layout, topology_inst, "mbox_priorbox", {
        {conv11, {60},  {},    {2.0}},
        {conv13, {105}, {150}, {2.0, 3.0}},
        {conv14, {150}, {195}, {2.0, 3.0}},
        {conv15, {195}, {240}, {2.0, 3.0}},
        {conv16, {240}, {285}, {2.0, 3.0}},
        {conv17, {285}, {300}, {2.0, 3.0}}
    });

    // Combining multi-box information into detection output.
    auto mbox_conf_reshape = reshape(
        "mbox_conf_reshape",
        mbox_conf,
        {batch_size, 1917, 21, 1}
    );
    auto mbox_conf_softmax = softmax(
        "mbox_conf_softmax",
        mbox_conf_reshape,
        softmax::normalize_x
    );
    auto mbox_conf_flatten = reshape(
        "mbox_conf_flatten",
        mbox_conf_softmax,
        {batch_size, 40257, 1, 1}
    );

    auto detection_out = detection_output(
        "output",
        mbox_loc,
        mbox_conf_flatten,
        mbox_priorbox,
        21,
        100,
        true,
        0,
        0.45f,
        100,
        1,
        prior_box_code_type::center_size,
        false,
        0.25f
    );

    topology_inst.add(
        mbox_conf_reshape, mbox_conf_softmax, mbox_conf_flatten,
        detection_out
    );
    return topology_inst;
}
