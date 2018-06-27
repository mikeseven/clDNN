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

using namespace cldnn;

void generate_wb_vectors(
    const std::string& weights_dir,
    const cldnn::engine& engine,
    const std::string& name,
    unsigned group_count,
    std::vector<cldnn::data>& weights_data,
    std::vector<primitive_id>& weithts_prim_id,
    std::vector<cldnn::data>& biases_data,
    std::vector<primitive_id>& biases_prim_id)
{
    for (unsigned i = 0; i < group_count; i++)
    {
        unsigned group_num = i + 1;

        auto weight = file::create({ engine, join_path(weights_dir, name + "_g" + std::to_string(group_num) + "_weights.nnd") });
        weights_data.push_back(weight);
        weithts_prim_id.push_back(weight);

        auto bias = file::create({ engine, join_path(weights_dir, name + "_g" + std::to_string(group_num) + "_bias.nnd") });
        biases_data.push_back(bias);
        biases_prim_id.push_back(bias);
    }
}

template<typename T>
void set_values(const cldnn::memory& mem, const int count) {
    auto ptr = mem.pointer<T>();

    auto it = ptr.begin();
    for (auto x : args)
        *it++ = x;
}

// Building SSD MobileNet network with loading weights & biases from file
cldnn::topology build_ssd_mobilenet(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& in_layout, int32_t batch_size)
{
    cldnn::topology topology;
    // [300x300x3xB]
    in_layout.size = { batch_size, 3, 300, 300 };
    auto input = cldnn::input_layout("input", in_layout);

    // subtract mean values
    //auto reordered_input = reorder(
    //    "reorder",
    //    input,
    //    { in_layout.data_type, in_layout.format, in_layout.size },
    //    std::vector<float>{ 104.0f, 117.0f, 123.0f });

    auto scale_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_ptr = scale_mem.pointer<float>();
    //for (int i = 0; i < scale_mem.get_layout().size; i++)
    {
        scale_ptr[0] = 0.01699986400108799;
    }
    auto scale_data = data("scale_data", scale_mem);

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

    auto conv0_w = file::create({ engine, join_path(weights_dir, "conv0_weights.nnd") });
    auto conv0_b = file::create({ engine, join_path(weights_dir, "conv0_bias.nnd") });
    auto conv0 = convolution(
        "conv0",
        mull_340,
        { conv0_w },
        { conv0_b },
        { 1,1,2,2 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv1_dw_w_data;
    std::vector<primitive_id> conv1_dw_w_prim_id;
    std::vector<cldnn::data> conv1_dw_b_data;
    std::vector<primitive_id> conv1_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv1%2Fdw", 32, conv1_dw_w_data, conv1_dw_w_prim_id, conv1_dw_b_data, conv1_dw_b_prim_id);
    auto conv1_dw = convolution(
        "conv1_dw",
        conv0,
        conv1_dw_w_prim_id,
        conv1_dw_b_prim_id,
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv1_w = file::create({ engine, join_path(weights_dir, "conv1_weights.nnd") });
    auto conv1_b = file::create({ engine, join_path(weights_dir, "conv1_bias.nnd") });
    auto conv1 = convolution(
        "conv1",
        conv1_dw,
        { conv1_w },
        { conv1_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv2_dw_w_data;
    std::vector<primitive_id> conv2_dw_w_prim_id;
    std::vector<cldnn::data> conv2_dw_b_data;
    std::vector<primitive_id> conv2_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv2%2Fdw", 64, conv2_dw_w_data, conv2_dw_w_prim_id, conv2_dw_b_data, conv2_dw_b_prim_id);
    auto conv2_dw = convolution(
        "conv2_dw",
        conv1,
        conv2_dw_w_prim_id,
        conv2_dw_b_prim_id,
        { 1,1,2,2 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv2_w = file::create({ engine, join_path(weights_dir, "conv2_weights.nnd") });
    auto conv2_b = file::create({ engine, join_path(weights_dir, "conv2_bias.nnd") });
    auto conv2 = convolution(
        "conv2",
        conv2_dw,
        { conv2_w },
        { conv2_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv3_dw_w_data;
    std::vector<primitive_id> conv3_dw_w_prim_id;
    std::vector<cldnn::data> conv3_dw_b_data;
    std::vector<primitive_id> conv3_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv3%2Fdw", 128, conv3_dw_w_data, conv3_dw_w_prim_id, conv3_dw_b_data, conv3_dw_b_prim_id);
    auto conv3_dw = convolution(
        "conv3_dw",
        conv2,
        conv3_dw_w_prim_id,
        conv3_dw_b_prim_id,
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv3_w = file::create({ engine, join_path(weights_dir, "conv3_weights.nnd") });
    auto conv3_b = file::create({ engine, join_path(weights_dir, "conv3_bias.nnd") });
    auto conv3 = convolution(
        "conv3",
        conv3_dw,
        { conv3_w },
        { conv3_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv4_dw_w_data;
    std::vector<primitive_id> conv4_dw_w_prim_id;
    std::vector<cldnn::data> conv4_dw_b_data;
    std::vector<primitive_id> conv4_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv4%2Fdw", 128, conv4_dw_w_data, conv4_dw_w_prim_id, conv4_dw_b_data, conv4_dw_b_prim_id);
    auto conv4_dw = convolution(
        "conv4_dw",
        conv3,
        conv4_dw_w_prim_id,
        conv4_dw_b_prim_id,
        { 1,1,2,2 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv4_w = file::create({ engine, join_path(weights_dir, "conv4_weights.nnd") });
    auto conv4_b = file::create({ engine, join_path(weights_dir, "conv4_bias.nnd") });
    auto conv4 = convolution(
        "conv4",
        conv4_dw,
        { conv4_w },
        { conv4_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv5_dw_w_data;
    std::vector<primitive_id> conv5_dw_w_prim_id;
    std::vector<cldnn::data> conv5_dw_b_data;
    std::vector<primitive_id> conv5_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv5%2Fdw", 256, conv5_dw_w_data, conv5_dw_w_prim_id, conv5_dw_b_data, conv5_dw_b_prim_id);
    auto conv5_dw = convolution(
        "conv5_dw",
        conv4,
        conv5_dw_w_prim_id,
        conv5_dw_b_prim_id,
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv5_w = file::create({ engine, join_path(weights_dir, "conv5_weights.nnd") });
    auto conv5_b = file::create({ engine, join_path(weights_dir, "conv5_bias.nnd") });
    auto conv5 = convolution(
        "conv5",
        conv5_dw,
        { conv5_w },
        { conv5_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv6_dw_w_data;
    std::vector<primitive_id> conv6_dw_w_prim_id;
    std::vector<cldnn::data> conv6_dw_b_data;
    std::vector<primitive_id> conv6_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv6%2Fdw", 256, conv6_dw_w_data, conv6_dw_w_prim_id, conv6_dw_b_data, conv6_dw_b_prim_id);
    auto conv6_dw = convolution(
        "conv6_dw",
        conv5,
        conv6_dw_w_prim_id,
        conv6_dw_b_prim_id,
        { 1,1,2,2 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv6_w = file::create({ engine, join_path(weights_dir, "conv6_weights.nnd") });
    auto conv6_b = file::create({ engine, join_path(weights_dir, "conv6_bias.nnd") });
    auto conv6 = convolution(
        "conv6",
        conv6_dw,
        { conv6_w },
        { conv6_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv7_dw_w_data;
    std::vector<primitive_id> conv7_dw_w_prim_id;
    std::vector<cldnn::data> conv7_dw_b_data;
    std::vector<primitive_id> conv7_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv7%2Fdw", 512, conv7_dw_w_data, conv7_dw_w_prim_id, conv7_dw_b_data, conv7_dw_b_prim_id);
    auto conv7_dw = convolution(
        "conv7_dw",
        conv6,
        conv7_dw_w_prim_id,
        conv7_dw_b_prim_id,
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv7_w = file::create({ engine, join_path(weights_dir, "conv7_weights.nnd") });
    auto conv7_b = file::create({ engine, join_path(weights_dir, "conv7_bias.nnd") });
    auto conv7 = convolution(
        "conv7",
        conv7_dw,
        { conv7_w },
        { conv7_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv8_dw_w_data;
    std::vector<primitive_id> conv8_dw_w_prim_id;
    std::vector<cldnn::data> conv8_dw_b_data;
    std::vector<primitive_id> conv8_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv8%2Fdw", 512, conv8_dw_w_data, conv8_dw_w_prim_id, conv8_dw_b_data, conv8_dw_b_prim_id);
    auto conv8_dw = convolution(
        "conv8_dw",
        conv7,
        conv8_dw_w_prim_id,
        conv8_dw_b_prim_id,
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv8_w = file::create({ engine, join_path(weights_dir, "conv8_weights.nnd") });
    auto conv8_b = file::create({ engine, join_path(weights_dir, "conv8_bias.nnd") });
    auto conv8 = convolution(
        "conv8",
        conv8_dw,
        { conv8_w },
        { conv8_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv9_dw_w_data;
    std::vector<primitive_id> conv9_dw_w_prim_id;
    std::vector<cldnn::data> conv9_dw_b_data;
    std::vector<primitive_id> conv9_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv9%2Fdw", 512, conv9_dw_w_data, conv9_dw_w_prim_id, conv9_dw_b_data, conv9_dw_b_prim_id);
    auto conv9_dw = convolution(
        "conv9_dw",
        conv8,
        conv9_dw_w_prim_id,
        conv9_dw_b_prim_id,
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv9_w = file::create({ engine, join_path(weights_dir, "conv9_weights.nnd") });
    auto conv9_b = file::create({ engine, join_path(weights_dir, "conv9_bias.nnd") });
    auto conv9 = convolution(
        "conv9",
        conv9_dw,
        { conv9_w },
        { conv9_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv10_dw_w_data;
    std::vector<primitive_id> conv10_dw_w_prim_id;
    std::vector<cldnn::data> conv10_dw_b_data;
    std::vector<primitive_id> conv10_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv10%2Fdw", 512, conv10_dw_w_data, conv10_dw_w_prim_id, conv10_dw_b_data, conv10_dw_b_prim_id);
    auto conv10_dw = convolution(
        "conv10_dw",
        conv9,
        conv10_dw_w_prim_id,
        conv10_dw_b_prim_id,
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv10_w = file::create({ engine, join_path(weights_dir, "conv10_weights.nnd") });
    auto conv10_b = file::create({ engine, join_path(weights_dir, "conv10_bias.nnd") });
    auto conv10 = convolution(
        "conv10",
        conv10_dw,
        { conv10_w },
        { conv10_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv11_dw_w_data;
    std::vector<primitive_id> conv11_dw_w_prim_id;
    std::vector<cldnn::data> conv11_dw_b_data;
    std::vector<primitive_id> conv11_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv11%2Fdw", 512, conv11_dw_w_data, conv11_dw_w_prim_id, conv11_dw_b_data, conv11_dw_b_prim_id);
    auto conv11_dw = convolution(
        "conv11_dw",
        conv10,
        conv11_dw_w_prim_id,
        conv11_dw_b_prim_id,
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv11_w = file::create({ engine, join_path(weights_dir, "conv11_weights.nnd") });
    auto conv11_b = file::create({ engine, join_path(weights_dir, "conv11_bias.nnd") });
    auto conv11 = convolution(
        "conv11",
        conv11_dw,
        { conv11_w },
        { conv11_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto conv11_mbox_priorbox = prior_box(
        "conv11_mbox_priorbox",
        conv11,
        in_layout.size,
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

    std::vector<cldnn::data> conv12_dw_w_data;
    std::vector<primitive_id> conv12_dw_w_prim_id;
    std::vector<cldnn::data> conv12_dw_b_data;
    std::vector<primitive_id> conv12_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv12%2Fdw", 512, conv12_dw_w_data, conv12_dw_w_prim_id, conv12_dw_b_data, conv12_dw_b_prim_id);
    auto conv12_dw = convolution(
        "conv12_dw",
        conv11,
        conv12_dw_w_prim_id,
        conv12_dw_b_prim_id,
        { 1,1,2,2 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv12_w = file::create({ engine, join_path(weights_dir, "conv12_weights.nnd") });
    auto conv12_b = file::create({ engine, join_path(weights_dir, "conv12_bias.nnd") });
    auto conv12 = convolution(
        "conv12",
        conv12_dw,
        { conv12_w },
        { conv12_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    std::vector<cldnn::data> conv13_dw_w_data;
    std::vector<primitive_id> conv13_dw_w_prim_id;
    std::vector<cldnn::data> conv13_dw_b_data;
    std::vector<primitive_id> conv13_dw_b_prim_id;
    generate_wb_vectors(weights_dir, engine, "conv13%2Fdw", 1024, conv13_dw_w_data, conv13_dw_w_prim_id, conv13_dw_b_data, conv13_dw_b_prim_id);
    auto conv13_dw = convolution(
        "conv13_dw",
        conv12,
        conv13_dw_w_prim_id,
        conv13_dw_b_prim_id,
        { 1,1,1,1 },
        { 0,0,-1,-1 },
        { 1,1,1,1 },
        true);

    auto conv13_w = file::create({ engine, join_path(weights_dir, "conv13_weights.nnd") });
    auto conv13_b = file::create({ engine, join_path(weights_dir, "conv13_bias.nnd") });
    auto conv13 = convolution(
        "conv13",
        conv13_dw,
        { conv13_w },
        { conv13_b },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto conv13_mbox_priorbox = prior_box(
        "conv13_mbox_priorbox",
        conv13,
        in_layout.size,
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
        in_layout.size,
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
        in_layout.size,
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
        in_layout.size,
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
        in_layout.size,
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

    topology.add(
        input,
        //reordered_input,
        scale_data,
        mull_340,
        conv0, conv0_w, conv0_b);

    topology.add(conv1_dw);
    for (unsigned i = 0; i < conv1_dw_w_data.size(); i++)
    {
        topology.add(conv1_dw_w_data.at(i), conv1_dw_b_data.at(i));
    }

    topology.add(conv1, conv1_w, conv1_b);

    topology.add(conv2_dw);
    for (unsigned i = 0; i < conv2_dw_w_data.size(); i++)
    {
        topology.add(conv2_dw_w_data.at(i), conv2_dw_b_data.at(i));
    }

    topology.add(conv2, conv2_w, conv2_b);

    topology.add(conv3_dw);
    for (unsigned i = 0; i < conv3_dw_w_data.size(); i++)
    {
        topology.add(conv3_dw_w_data.at(i), conv3_dw_b_data.at(i));
    }

    topology.add(conv3, conv3_w, conv3_b);

    topology.add(conv4_dw);
    for (unsigned i = 0; i < conv4_dw_w_data.size(); i++)
    {
        topology.add(conv4_dw_w_data.at(i), conv4_dw_b_data.at(i));
    }

    topology.add(conv4, conv4_w, conv4_b);

    topology.add(conv5_dw); 
    for (unsigned i = 0; i < conv5_dw_w_data.size(); i++)
    {
        topology.add(conv5_dw_w_data.at(i), conv5_dw_b_data.at(i));
    }

    topology.add(conv5, conv5_w, conv5_b);

    topology.add(conv6_dw);
    for (unsigned i = 0; i < conv6_dw_w_data.size(); i++)
    {
        topology.add(conv6_dw_w_data.at(i), conv6_dw_b_data.at(i));
    }

    topology.add(conv6, conv6_w, conv6_b);

    topology.add(conv7_dw);
    for (unsigned i = 0; i < conv7_dw_w_data.size(); i++)
    {
        topology.add(conv7_dw_w_data.at(i), conv7_dw_b_data.at(i));
    }

    topology.add(conv7, conv7_w, conv7_b);

    topology.add(conv8_dw);
    for (unsigned i = 0; i < conv8_dw_w_data.size(); i++)
    {
        topology.add(conv8_dw_w_data.at(i), conv8_dw_b_data.at(i));
    }

    topology.add(conv8, conv8_w, conv8_b);

    topology.add(conv9_dw);
    for (unsigned i = 0; i < conv9_dw_w_data.size(); i++)
    {
        topology.add(conv9_dw_w_data.at(i), conv9_dw_b_data.at(i));
    }

    topology.add(conv9, conv9_w, conv9_b);

    topology.add(conv10_dw);
    for (unsigned i = 0; i < conv10_dw_w_data.size(); i++)
    {
        topology.add(conv10_dw_w_data.at(i), conv10_dw_b_data.at(i));
    }

    topology.add(conv10, conv10_w, conv10_b);

    topology.add(conv11_dw);
    for (unsigned i = 0; i < conv11_dw_w_data.size(); i++)
    {
        topology.add(conv11_dw_w_data.at(i), conv11_dw_b_data.at(i));
    }

    topology.add(conv11, conv11_w, conv11_b);

    topology.add(
        conv11_mbox_priorbox
    );

    topology.add(
        conv11_mbox_conf, conv11_mbox_conf_w, conv11_mbox_conf_b,
        conv11_mbox_conf_perm,
        conv11_mbox_conf_flat
    );

    topology.add(
        conv11_mbox_loc, conv11_mbox_loc_w, conv11_mbox_loc_b,
        conv11_mbox_loc_perm,
        conv11_mbox_loc_flat
    );

    topology.add(conv12_dw);
    for (unsigned i = 0; i < conv12_dw_w_data.size(); i++)
    {
        topology.add(conv12_dw_w_data.at(i), conv12_dw_b_data.at(i));
    }

    topology.add(conv12, conv12_w, conv12_b);

    topology.add(conv13_dw);
    for (unsigned i = 0; i < conv13_dw_w_data.size(); i++)
    {
        topology.add(conv13_dw_w_data.at(i), conv13_dw_b_data.at(i));
    }

    topology.add(conv13, conv13_w, conv13_b);

    topology.add(
        conv13_mbox_priorbox
    );

    topology.add(
        conv13_mbox_conf, conv13_mbox_conf_w, conv13_mbox_conf_b,
        conv13_mbox_conf_perm,
        conv13_mbox_conf_flat
    );

    topology.add(
        conv13_mbox_loc, conv13_mbox_loc_w, conv13_mbox_loc_b,
        conv13_mbox_loc_perm,
        conv13_mbox_loc_flat
    );

    topology.add(
        conv14_1, conv14_1_w, conv14_1_b,
        conv14_2, conv14_2_w, conv14_2_b
    );

    topology.add(
        conv14_2_mbox_priorbox
    );

    topology.add(
        conv14_2_mbox_conf, conv14_2_mbox_conf_w, conv14_2_mbox_conf_b,
        conv14_2_mbox_conf_perm,
        conv14_2_mbox_conf_flat
    );

    topology.add(
        conv14_2_mbox_loc, conv14_2_mbox_loc_w, conv14_2_mbox_loc_b,
        conv14_2_mbox_loc_perm,
        conv14_2_mbox_loc_flat
    );

    topology.add(
        conv15_1, conv15_1_w, conv15_1_b,
        conv15_2, conv15_2_w, conv15_2_b
    );

    topology.add(
        conv15_2_mbox_priorbox
    );

    topology.add(
        conv15_2_mbox_conf, conv15_2_mbox_conf_w, conv15_2_mbox_conf_b,
        conv15_2_mbox_conf_perm,
        conv15_2_mbox_conf_flat
    );

    topology.add(
        conv15_2_mbox_loc, conv15_2_mbox_loc_w, conv15_2_mbox_loc_b,
        conv15_2_mbox_loc_perm,
        conv15_2_mbox_loc_flat
    );

    topology.add(
        conv16_1, conv16_1_w, conv16_1_b,
        conv16_2, conv16_2_w, conv16_2_b
    );

    topology.add(
        conv16_2_mbox_priorbox
    );

    topology.add(
        conv16_2_mbox_conf, conv16_2_mbox_conf_w, conv16_2_mbox_conf_b,
        conv16_2_mbox_conf_perm,
        conv16_2_mbox_conf_flat
    );

    topology.add(
        conv16_2_mbox_loc, conv16_2_mbox_loc_w, conv16_2_mbox_loc_b,
        conv16_2_mbox_loc_perm,
        conv16_2_mbox_loc_flat
    );

    topology.add(
        conv17_1, conv17_1_w, conv17_1_b,
        conv17_2, conv17_2_w, conv17_2_b
    );

    topology.add(
        conv17_2_mbox_priorbox
    );

    topology.add(
        conv17_2_mbox_conf, conv17_2_mbox_conf_w, conv17_2_mbox_conf_b,
        conv17_2_mbox_conf_perm,
        conv17_2_mbox_conf_flat
    );

    topology.add(
        conv17_2_mbox_loc, conv17_2_mbox_loc_w, conv17_2_mbox_loc_b,
        conv17_2_mbox_loc_perm,
        conv17_2_mbox_loc_flat
    );

    topology.add(
        mbox_priorbox,
        mbox_loc,
        mbox_conf,
        mbox_conf_reshape,
        mbox_conf_softmax,
        mbox_conf_flatten,
        detection_out
    );

    return topology;
}
