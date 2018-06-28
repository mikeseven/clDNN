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

#include <api/CPP/input_layout.hpp>
#include <api/CPP/reorder.hpp>
#include <api/CPP/eltwise.hpp>
#include <api/CPP/scale.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/lrn.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/concatenation.hpp>
#include <api/CPP/softmax.hpp>
#include <api/CPP/prior_box.hpp>
#include <api/CPP/permute.hpp>
#include <api/CPP/reshape.hpp>
#include <api/CPP/detection_output.hpp>

#include "common/common_tools.h"
#include "file.h"

#include <string>


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
            auto weights_data = file::create({engine, join_path(weights_dir, weights_root + "_g" + std::to_string(gi) + "_weights.nnd")});
            auto bias_data    = file::create({engine, join_path(weights_dir, weights_root + "_g" + std::to_string(gi) + "_bias.nnd")});

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
        add_relu);

    topology_inst.add(conv_layer);

    return conv_layer;
}



// Building SSD MobileNet network with loading weights & biases from file
cldnn::topology build_ssd_mobilenet(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    // [300x300x3xB]
    input_layout.size = {batch_size, 3, 300, 300};
    auto input        = cldnn::input_layout("input", input_layout);
    cldnn::topology topology_inst{input};

    // subtract mean values
    //auto reordered_input = reorder(
    //    "reorder",
    //    input,
    //    { input_layout.data_type, input_layout.format, input_layout.size },
    //    std::vector<float>{ 104.0f, 117.0f, 123.0f });

    auto scale_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_ptr = scale_mem.pointer<float>();
    //for (int i = 0; i < scale_mem.get_layout().size; i++)
    {
        scale_ptr[0] = 0.01699986400108799f;
    }
    auto scale_data = cldnn::data("scale_data", scale_mem);

    ///*auto eltwise0 = eltwise(
    //    "eltwise0",
    //    { input },
    //    { "scale_data" },
    //    eltwise_mode::prod
    //);*/

    auto mull_340 = scale(
        "mull_340",
        input,
        scale_data
    );

    auto conv0 = add_conv_layer(weights_dir, engine, topology_inst, "conv0", "conv0", mull_340, {0, 0, -1, -1}, {1, 1, 2, 2});

    auto conv1_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv1_dw", "conv1%2Fdw", conv0, {0, 0, -1, -1}, {1, 1, 1, 1}, 32);
    auto conv1    = add_conv_layer(weights_dir, engine, topology_inst, "conv1", "conv1", conv1_dw);

    auto conv2_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv2_dw", "conv2%2Fdw", conv1, {0, 0, -1, -1}, {1, 1, 2, 2}, 64);
    auto conv2    = add_conv_layer(weights_dir, engine, topology_inst, "conv2", "conv2", conv2_dw);

    auto conv3_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv3_dw", "conv3%2Fdw", conv2, {0, 0, -1, -1}, {1, 1, 1, 1}, 128);
    auto conv3    = add_conv_layer(weights_dir, engine, topology_inst, "conv3", "conv3", conv3_dw);

    auto conv4_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv4_dw", "conv4%2Fdw", conv3, {0, 0, -1, -1}, {1, 1, 2, 2}, 128);
    auto conv4    = add_conv_layer(weights_dir, engine, topology_inst, "conv4", "conv4", conv4_dw);

    auto conv5_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv5_dw", "conv5%2Fdw", conv4, {0, 0, -1, -1}, {1, 1, 1, 1}, 256);
    auto conv5    = add_conv_layer(weights_dir, engine, topology_inst, "conv5", "conv5", conv5_dw);

    auto conv6_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv6_dw", "conv6%2Fdw", conv5, {0, 0, -1, -1}, {1, 1, 2, 2}, 256);
    auto conv6    = add_conv_layer(weights_dir, engine, topology_inst, "conv6", "conv6", conv6_dw);

    auto conv7_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv7_dw", "conv7%2Fdw", conv6, {0, 0, -1, -1}, {1, 1, 1, 1}, 512);
    auto conv7    = add_conv_layer(weights_dir, engine, topology_inst, "conv7", "conv7", conv7_dw);

    auto conv8_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv8_dw", "conv8%2Fdw", conv7, {0, 0, -1, -1}, {1, 1, 1, 1}, 512);
    auto conv8    = add_conv_layer(weights_dir, engine, topology_inst, "conv8", "conv8", conv8_dw);

    auto conv9_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv9_dw", "conv9%2Fdw", conv8, {0, 0, -1, -1}, {1, 1, 1, 1}, 512);
    auto conv9    = add_conv_layer(weights_dir, engine, topology_inst, "conv9", "conv9", conv9_dw);

    auto conv10_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv10_dw", "conv10%2Fdw", conv9, {0, 0, -1, -1}, {1, 1, 1, 1}, 512);
    auto conv10    = add_conv_layer(weights_dir, engine, topology_inst, "conv10", "conv10", conv10_dw);

    auto conv11_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv11_dw", "conv11%2Fdw", conv10, {0, 0, -1, -1}, {1, 1, 1, 1}, 512);
    auto conv11    = add_conv_layer(weights_dir, engine, topology_inst, "conv11", "conv11", conv11_dw);

    auto conv12_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv12_dw", "conv12%2Fdw", conv11, {0, 0, -1, -1}, {1, 1, 2, 2}, 512);
    auto conv12    = add_conv_layer(weights_dir, engine, topology_inst, "conv12", "conv12", conv12_dw);

    auto conv13_dw = add_conv_layer(weights_dir, engine, topology_inst, "conv13_dw", "conv13%2Fdw", conv12, {0, 0, -1, -1}, {1, 1, 1, 1}, 1024);
    auto conv13    = add_conv_layer(weights_dir, engine, topology_inst, "conv13", "conv13", conv13_dw);


    auto conv11_mbox_priorbox = prior_box(
        "conv11_mbox_priorbox",
        conv11,
        input_layout.size,
        { 60 },
        {},
        { 2 },
        true,
        false,
        { 0.1f,0.1f,0.2f,0.2f },
        0,
        0,
        0.5f);

    auto conv11_mbox_conf_w = file::create({ engine, join_path(weights_dir, "conv11_mbox_conf_weights.nnd") });
    auto conv11_mbox_conf_b = file::create({ engine, join_path(weights_dir, "conv11_mbox_conf_bias.nnd") });
    auto conv11_mbox_conf = convolution(
        "conv11_mbox_conf",
        conv11,
        { conv11_mbox_conf_w },
        { conv11_mbox_conf_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv11_mbox_conf_perm = permute(
        "conv11_mbox_conf_perm",
        conv11_mbox_conf,
        { 0,2,3,1 });

    auto conv11_mbox_conf_flat = reshape(
        "conv11_mbox_conf_flat",
        conv11_mbox_conf_perm,
        { batch_size, 22743,1,1 });

    auto conv11_mbox_loc_w = file::create({ engine, join_path(weights_dir, "conv11_mbox_loc_weights.nnd") });
    auto conv11_mbox_loc_b = file::create({ engine, join_path(weights_dir, "conv11_mbox_loc_bias.nnd") });
    auto conv11_mbox_loc = convolution(
        "conv11_mbox_loc",
        conv11,
        { conv11_mbox_loc_w },
        { conv11_mbox_loc_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv11_mbox_loc_perm = permute(
        "conv11_mbox_loc_perm",
        conv11_mbox_loc,
        { 0,2,3,1 });

    auto conv11_mbox_loc_flat = reshape(
        "conv11_mbox_loc_flat",
        conv11_mbox_loc_perm,
        { batch_size,4332,1,1 });

    auto conv13_mbox_priorbox = prior_box(
        "conv13_mbox_priorbox",
        conv13,
        input_layout.size,
        { 105 },
        { 150 },
        { 2.0,3.0 },
        true,
        false,
        { 0.1f,0.1f,0.2f,0.2f },
        0,
        0,
        0.5f);

    auto conv13_mbox_conf_w = file::create({ engine, join_path(weights_dir, "conv13_mbox_conf_weights.nnd") });
    auto conv13_mbox_conf_b = file::create({ engine, join_path(weights_dir, "conv13_mbox_conf_bias.nnd") });
    auto conv13_mbox_conf = convolution(
        "conv13_mbox_conf",
        conv13,
        { conv13_mbox_conf_w },
        { conv13_mbox_conf_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv13_mbox_conf_perm = permute(
        "conv13_mbox_conf_perm",
        conv13_mbox_conf,
        { 0,2,3,1 });

    auto conv13_mbox_conf_flat = reshape(
        "conv13_mbox_conf_flat",
        conv13_mbox_conf_perm,
        { batch_size,12600,1,1 });

    auto conv13_mbox_loc_w = file::create({ engine, join_path(weights_dir, "conv13_mbox_loc_weights.nnd") });
    auto conv13_mbox_loc_b = file::create({ engine, join_path(weights_dir, "conv13_mbox_loc_bias.nnd") });
    auto conv13_mbox_loc = convolution(
        "conv13_mbox_loc",
        conv13,
        { conv13_mbox_loc_w },
        { conv13_mbox_loc_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv13_mbox_loc_perm = permute(
        "conv13_mbox_loc_perm",
        conv13_mbox_loc,
        { 0,2,3,1 });

    auto conv13_mbox_loc_flat = reshape(
        "conv13_mbox_loc_flat",
        conv13_mbox_loc_perm,
        { batch_size,2400,1,1 });

    auto conv14_1_w = file::create({ engine, join_path(weights_dir, "conv14_1_weights.nnd") });
    auto conv14_1_b = file::create({ engine, join_path(weights_dir, "conv14_1_bias.nnd") });
    auto conv14_1 = convolution(
        "conv14_1",
        conv13,
        { conv14_1_w },
        { conv14_1_b },
        { 1,1,1,1 },
        { 0,0,0,-0 },
        { 1,1,1,1 },
        true);

    auto conv14_2_w = file::create({ engine, join_path(weights_dir, "conv14_2_weights.nnd") });
    auto conv14_2_b = file::create({ engine, join_path(weights_dir, "conv14_2_bias.nnd") });
    auto conv14_2 = convolution(
        "conv14_2",
        conv14_1,
        { conv14_2_w },
        { conv14_2_b },
        { 1,1,2,2 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv14_2_mbox_priorbox = prior_box(
        "conv14_2_mbox_priorbox",
        conv14_2,
        input_layout.size,
        { 150 },
        { 195 },
        { 2.0,3.0 },
        true,
        false,
        { 0.1f,0.1f,0.2f,0.2f },
        0,
        0,
        0.5f);

    auto conv14_2_mbox_conf_w = file::create({ engine, join_path(weights_dir, "conv14_2_mbox_conf_weights.nnd") });
    auto conv14_2_mbox_conf_b = file::create({ engine, join_path(weights_dir, "conv14_2_mbox_conf_bias.nnd") });
    auto conv14_2_mbox_conf = convolution(
        "conv14_2_mbox_conf",
        conv14_2,
        { conv14_2_mbox_conf_w },
        { conv14_2_mbox_conf_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv14_2_mbox_conf_perm = permute(
        "conv14_2_mbox_conf_perm",
        conv14_2_mbox_conf,
        { 0,2,3,1 });

    auto conv14_2_mbox_conf_flat = reshape(
        "conv14_2_mbox_conf_flat",
        conv14_2_mbox_conf_perm,
        { batch_size,3150,1,1 });

    auto conv14_2_mbox_loc_w = file::create({ engine, join_path(weights_dir, "conv14_2_mbox_loc_weights.nnd") });
    auto conv14_2_mbox_loc_b = file::create({ engine, join_path(weights_dir, "conv14_2_mbox_loc_bias.nnd") });
    auto conv14_2_mbox_loc = convolution(
        "conv14_2_mbox_loc",
        conv14_2,
        { conv14_2_mbox_loc_w },
        { conv14_2_mbox_loc_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv14_2_mbox_loc_perm = permute(
        "conv14_2_mbox_loc_perm",
        conv14_2_mbox_loc,
        { 0,2,3,1 });

    auto conv14_2_mbox_loc_flat = reshape(
        "conv14_2_mbox_loc_flat",
        conv14_2_mbox_loc_perm,
        { batch_size,600,1,1 });

    auto conv15_1_w = file::create({ engine, join_path(weights_dir, "conv15_1_weights.nnd") });
    auto conv15_1_b = file::create({ engine, join_path(weights_dir, "conv15_1_bias.nnd") });
    auto conv15_1 = convolution(
        "conv15_1",
        conv14_2,
        { conv15_1_w },
        { conv15_1_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto conv15_2_w = file::create({ engine, join_path(weights_dir, "conv15_2_weights.nnd") });
    auto conv15_2_b = file::create({ engine, join_path(weights_dir, "conv15_2_bias.nnd") });
    auto conv15_2 = convolution(
        "conv15_2",
        conv15_1,
        { conv15_2_w },
        { conv15_2_b },
        { 1,1,2,2 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv15_2_mbox_priorbox = prior_box(
        "conv15_2_mbox_priorbox",
        conv15_2,
        input_layout.size,
        { 195 },
        { 240 },
        { 2.0,3.0 },
        true,
        false,
        { 0.1f,0.1f,0.2f,0.2f },
        0,
        0,
        0.5f);

    auto conv15_2_mbox_conf_w = file::create({ engine, join_path(weights_dir, "conv15_2_mbox_conf_weights.nnd") });
    auto conv15_2_mbox_conf_b = file::create({ engine, join_path(weights_dir, "conv15_2_mbox_conf_bias.nnd") });
    auto conv15_2_mbox_conf = convolution(
        "conv15_2_mbox_conf",
        conv15_2,
        { conv15_2_mbox_conf_w },
        { conv15_2_mbox_conf_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv15_2_mbox_conf_perm = permute(
        "conv15_2_mbox_conf_perm",
        conv15_2_mbox_conf,
        { 0,2,3,1 });

    auto conv15_2_mbox_conf_flat = reshape(
        "conv15_2_mbox_conf_flat",
        conv15_2_mbox_conf_perm,
        { batch_size,1134,1,1 });

    auto conv15_2_mbox_loc_w = file::create({ engine, join_path(weights_dir, "conv15_2_mbox_loc_weights.nnd") });
    auto conv15_2_mbox_loc_b = file::create({ engine, join_path(weights_dir, "conv15_2_mbox_loc_bias.nnd") });
    auto conv15_2_mbox_loc = convolution(
        "conv15_2_mbox_loc",
        conv15_2,
        { conv15_2_mbox_loc_w },
        { conv15_2_mbox_loc_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv15_2_mbox_loc_perm = permute(
        "conv15_2_mbox_loc_perm",
        conv15_2_mbox_loc,
        { 0,2,3,1 });

    auto conv15_2_mbox_loc_flat = reshape(
        "conv15_2_mbox_loc_flat",
        conv15_2_mbox_loc_perm,
        { batch_size,216,1,1 });

    auto conv16_1_w = file::create({ engine, join_path(weights_dir, "conv16_1_weights.nnd") });
    auto conv16_1_b = file::create({ engine, join_path(weights_dir, "conv16_1_bias.nnd") });
    auto conv16_1 = convolution(
        "conv16_1",
        conv15_2,
        { conv16_1_w },
        { conv16_1_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto conv16_2_w = file::create({ engine, join_path(weights_dir, "conv16_2_weights.nnd") });
    auto conv16_2_b = file::create({ engine, join_path(weights_dir, "conv16_2_bias.nnd") });
    auto conv16_2 = convolution(
        "conv16_2",
        conv16_1,
        { conv16_2_w },
        { conv16_2_b },
        { 1,1,2,2 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv16_2_mbox_priorbox = prior_box(
        "conv16_2_mbox_priorbox",
        conv16_2,
        input_layout.size,
        { 240 },
        { 285 },
        { 2.0,3.0 },
        true,
        false,
        { 0.1f,0.1f,0.2f,0.2f },
        0,
        0,
        0.5f);

    auto conv16_2_mbox_conf_w = file::create({ engine, join_path(weights_dir, "conv16_2_mbox_conf_weights.nnd") });
    auto conv16_2_mbox_conf_b = file::create({ engine, join_path(weights_dir, "conv16_2_mbox_conf_bias.nnd") });
    auto conv16_2_mbox_conf = convolution(
        "conv16_2_mbox_conf",
        conv16_2,
        { conv16_2_mbox_conf_w },
        { conv16_2_mbox_conf_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv16_2_mbox_conf_perm = permute(
        "conv16_2_mbox_conf_perm",
        conv16_2_mbox_conf,
        { 0,2,3,1 });

    auto conv16_2_mbox_conf_flat = reshape(
        "conv16_2_mbox_conf_flat",
        conv16_2_mbox_conf_perm,
        { batch_size,504,1,1 });

    auto conv16_2_mbox_loc_w = file::create({ engine, join_path(weights_dir, "conv16_2_mbox_loc_weights.nnd") });
    auto conv16_2_mbox_loc_b = file::create({ engine, join_path(weights_dir, "conv16_2_mbox_loc_bias.nnd") });
    auto conv16_2_mbox_loc = convolution(
        "conv16_2_mbox_loc",
        conv16_2,
        { conv16_2_mbox_loc_w },
        { conv16_2_mbox_loc_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv16_2_mbox_loc_perm = permute(
        "conv16_2_mbox_loc_perm",
        conv16_2_mbox_loc,
        { 0,2,3,1 });

    auto conv16_2_mbox_loc_flat = reshape(
        "conv16_2_mbox_loc_flat",
        conv16_2_mbox_loc_perm,
        { batch_size,96,1,1 });

    auto conv17_1_w = file::create({ engine, join_path(weights_dir, "conv17_1_weights.nnd") });
    auto conv17_1_b = file::create({ engine, join_path(weights_dir, "conv17_1_bias.nnd") });
    auto conv17_1 = convolution(
        "conv17_1",
        conv16_2,
        { conv17_1_w },
        { conv17_1_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto conv17_2_w = file::create({ engine, join_path(weights_dir, "conv17_2_weights.nnd") });
    auto conv17_2_b = file::create({ engine, join_path(weights_dir, "conv17_2_bias.nnd") });
    auto conv17_2 = convolution(
        "conv17_2",
        conv17_1,
        { conv17_2_w },
        { conv17_2_b },
        { 1,1,2,2 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv17_2_mbox_priorbox = prior_box(
        "conv17_2_mbox_priorbox",
        conv17_2,
        input_layout.size,
        { 285 },
        { 300 },
        { 2.0,3.0 },
        true,
        false,
        { 0.1f,0.1f,0.2f,0.2f },
        0,
        0,
        0.5f);

    auto conv17_2_mbox_conf_w = file::create({ engine, join_path(weights_dir, "conv17_2_mbox_conf_weights.nnd") });
    auto conv17_2_mbox_conf_b = file::create({ engine, join_path(weights_dir, "conv17_2_mbox_conf_bias.nnd") });
    auto conv17_2_mbox_conf = convolution(
        "conv17_2_mbox_conf",
        conv17_2,
        { conv17_2_mbox_conf_w },
        { conv17_2_mbox_conf_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv17_2_mbox_conf_perm = permute(
        "conv17_2_mbox_conf_perm",
        conv17_2_mbox_conf,
        { 0,2,3,1 });

    auto conv17_2_mbox_conf_flat = reshape(
        "conv17_2_mbox_conf_flat",
        conv17_2_mbox_conf_perm,
        { batch_size,126,1,1 });

    auto conv17_2_mbox_loc_w = file::create({ engine, join_path(weights_dir, "conv17_2_mbox_loc_weights.nnd") });
    auto conv17_2_mbox_loc_b = file::create({ engine, join_path(weights_dir, "conv17_2_mbox_loc_bias.nnd") });
    auto conv17_2_mbox_loc = convolution(
        "conv17_2_mbox_loc",
        conv17_2,
        { conv17_2_mbox_loc_w },
        { conv17_2_mbox_loc_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        false);

    auto conv17_2_mbox_loc_perm = permute(
        "conv17_2_mbox_loc_perm",
        conv17_2_mbox_loc,
        { 0,2,3,1 });

    auto conv17_2_mbox_loc_flat = reshape(
        "conv17_2_mbox_loc_flat",
        conv17_2_mbox_loc_perm,
        { batch_size,24,1,1 });

    auto mbox_priorbox = concatenation(
        "mbox_priorbox",
        {
            conv11_mbox_priorbox,
            conv13_mbox_priorbox,
            conv14_2_mbox_priorbox,
            conv15_2_mbox_priorbox,
            conv16_2_mbox_priorbox,
            conv17_2_mbox_priorbox
        },
        concatenation::along_y
    );

    auto mbox_conf = concatenation(
        "mbox_conf",
        {
            conv11_mbox_conf_flat,
            conv13_mbox_conf_flat,
            conv14_2_mbox_conf_flat,
            conv15_2_mbox_conf_flat,
            conv16_2_mbox_conf_flat,
            conv17_2_mbox_conf_flat
        },
        concatenation::along_f
    );

    auto mbox_conf_reshape = reshape(
        "mbox_conf_reshape",
        mbox_conf,
        { batch_size,1917,21,1}
    );

    auto mbox_conf_softmax = softmax(
        "mbox_conf_softmax",
        mbox_conf_reshape,
        cldnn::softmax::normalize_x
    );

    auto mbox_conf_flatten = reshape(
        "mbox_conf_flatten",
        mbox_conf_softmax,
        { batch_size,40257,1,1 });

    auto mbox_loc = concatenation(
        "mbox_loc",
        {
            conv11_mbox_loc_flat,
            conv13_mbox_loc_flat,
            conv14_2_mbox_loc_flat,
            conv15_2_mbox_loc_flat,
            conv16_2_mbox_loc_flat,
            conv17_2_mbox_loc_flat
        },
        concatenation::along_f
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
        input,
        //reordered_input,
        scale_data,
        mull_340);

    topology_inst.add(
        conv11_mbox_priorbox
    );

    topology_inst.add(
        conv11_mbox_conf, conv11_mbox_conf_w, conv11_mbox_conf_b,
        conv11_mbox_conf_perm,
        conv11_mbox_conf_flat
    );

    topology_inst.add(
        conv11_mbox_loc, conv11_mbox_loc_w, conv11_mbox_loc_b,
        conv11_mbox_loc_perm,
        conv11_mbox_loc_flat
    );

    topology_inst.add(
        conv13_mbox_priorbox
    );

    topology_inst.add(
        conv13_mbox_conf, conv13_mbox_conf_w, conv13_mbox_conf_b,
        conv13_mbox_conf_perm,
        conv13_mbox_conf_flat
    );

    topology_inst.add(
        conv13_mbox_loc, conv13_mbox_loc_w, conv13_mbox_loc_b,
        conv13_mbox_loc_perm,
        conv13_mbox_loc_flat
    );

    topology_inst.add(
        conv14_1, conv14_1_w, conv14_1_b,
        conv14_2, conv14_2_w, conv14_2_b
    );

    topology_inst.add(
        conv14_2_mbox_priorbox
    );

    topology_inst.add(
        conv14_2_mbox_conf, conv14_2_mbox_conf_w, conv14_2_mbox_conf_b,
        conv14_2_mbox_conf_perm,
        conv14_2_mbox_conf_flat
    );

    topology_inst.add(
        conv14_2_mbox_loc, conv14_2_mbox_loc_w, conv14_2_mbox_loc_b,
        conv14_2_mbox_loc_perm,
        conv14_2_mbox_loc_flat
    );

    topology_inst.add(
        conv15_1, conv15_1_w, conv15_1_b,
        conv15_2, conv15_2_w, conv15_2_b
    );

    topology_inst.add(
        conv15_2_mbox_priorbox
    );

    topology_inst.add(
        conv15_2_mbox_conf, conv15_2_mbox_conf_w, conv15_2_mbox_conf_b,
        conv15_2_mbox_conf_perm,
        conv15_2_mbox_conf_flat
    );

    topology_inst.add(
        conv15_2_mbox_loc, conv15_2_mbox_loc_w, conv15_2_mbox_loc_b,
        conv15_2_mbox_loc_perm,
        conv15_2_mbox_loc_flat
    );

    topology_inst.add(
        conv16_1, conv16_1_w, conv16_1_b,
        conv16_2, conv16_2_w, conv16_2_b
    );

    topology_inst.add(
        conv16_2_mbox_priorbox
    );

    topology_inst.add(
        conv16_2_mbox_conf, conv16_2_mbox_conf_w, conv16_2_mbox_conf_b,
        conv16_2_mbox_conf_perm,
        conv16_2_mbox_conf_flat
    );

    topology_inst.add(
        conv16_2_mbox_loc, conv16_2_mbox_loc_w, conv16_2_mbox_loc_b,
        conv16_2_mbox_loc_perm,
        conv16_2_mbox_loc_flat
    );

    topology_inst.add(
        conv17_1, conv17_1_w, conv17_1_b,
        conv17_2, conv17_2_w, conv17_2_b
    );

    topology_inst.add(
        conv17_2_mbox_priorbox
    );

    topology_inst.add(
        conv17_2_mbox_conf, conv17_2_mbox_conf_w, conv17_2_mbox_conf_b,
        conv17_2_mbox_conf_perm,
        conv17_2_mbox_conf_flat
    );

    topology_inst.add(
        conv17_2_mbox_loc, conv17_2_mbox_loc_w, conv17_2_mbox_loc_b,
        conv17_2_mbox_loc_perm,
        conv17_2_mbox_loc_flat
    );

    topology_inst.add(
        mbox_priorbox,
        mbox_loc,
        mbox_conf,
        mbox_conf_reshape,
        mbox_conf_softmax,
        mbox_conf_flatten,
        detection_out
    );

    return topology_inst;
}
