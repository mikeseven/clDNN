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
#include <api/CPP/convolution.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/fully_connected.hpp>
#include <api/CPP/softmax.hpp>

using namespace cldnn;

// Building vgg16 network with loading weights & biases from file
cldnn::topology build_vgg16(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { batch_size, 3, 224, 224 };
    auto input = cldnn::input_layout("input", input_layout);

    // subtract mean values
    auto reorder_mean = file::create({ engine, join_path(weights_dir, "imagenet_mean.nnd")});
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, input_layout.format, input_layout.size },
        reorder_mean);

    auto conv1_1_w = file::create({ engine, join_path(weights_dir, "conv1_1_weights.nnd")});
    auto conv1_1_b = file::create({ engine, join_path(weights_dir, "conv1_1_bias.nnd")});
    auto conv1_1 = convolution("conv1_1",
        reordered_input,
        { conv1_1_w },
        { conv1_1_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto conv1_2_w = file::create({ engine, join_path(weights_dir, "conv1_2_weights.nnd")});
    auto conv1_2_b = file::create({ engine, join_path(weights_dir, "conv1_2_bias.nnd") });
    auto conv1_2 = convolution("conv1_2",
            conv1_1,
        { conv1_2_w },
        { conv1_2_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU


    auto pool1 = pooling("pool1",
        conv1_2,
        pooling_mode::max,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv2_1_w = file::create({ engine, join_path(weights_dir, "conv2_1_weights.nnd")});
    auto conv2_1_b = file::create({ engine, join_path(weights_dir, "conv2_1_bias.nnd") });
    auto conv2_1 = convolution("conv2_1",
        pool1,
        { conv2_1_w },
        { conv2_1_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto conv2_2_w = file::create({ engine, join_path(weights_dir, "conv2_2_weights.nnd")});
    auto conv2_2_b = file::create({ engine, join_path(weights_dir, "conv2_2_bias.nnd") });
    auto conv2_2 = convolution("conv2_2",
        conv2_1,
        { conv2_2_w },
        { conv2_2_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto pool2 = pooling("pool2",
        conv2_2,
        pooling_mode::max,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv3_1_w = file::create({ engine, join_path(weights_dir, "conv3_1_weights.nnd")});
    auto conv3_1_b = file::create({ engine, join_path(weights_dir, "conv3_1_bias.nnd") });
    auto conv3_1 = convolution("conv3_1",
        pool2,
        { conv3_1_w },
        { conv3_1_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto conv3_2_w = file::create({ engine, join_path(weights_dir, "conv3_2_weights.nnd")});
    auto conv3_2_b = file::create({ engine, join_path(weights_dir, "conv3_2_bias.nnd") });
    auto conv3_2 = convolution("conv3_2",
        conv3_1,
        { conv3_2_w },
        { conv3_2_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto conv3_3_w = file::create({ engine, join_path(weights_dir, "conv3_3_weights.nnd")});
    auto conv3_3_b = file::create({ engine, join_path(weights_dir, "conv3_3_bias.nnd") });
    auto conv3_3 = convolution("conv3_3",
        conv3_2,
        { conv3_3_w },
        { conv3_3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto pool3 = pooling("pool3",
        conv3_3,
        pooling_mode::max,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv4_1_w = file::create({ engine, join_path(weights_dir, "conv4_1_weights.nnd")});
    auto conv4_1_b = file::create({ engine, join_path(weights_dir, "conv4_1_bias.nnd") });
    auto conv4_1 = convolution("conv4_1",
        pool3,
        { conv4_1_w },
        { conv4_1_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto conv4_2_w = file::create({ engine, join_path(weights_dir, "conv4_2_weights.nnd")});
    auto conv4_2_b = file::create({ engine, join_path(weights_dir, "conv4_2_bias.nnd") });
    auto conv4_2 = convolution("conv4_2",
        conv4_1,
        { conv4_2_w },
        { conv4_2_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto conv4_3_w = file::create({ engine, join_path(weights_dir, "conv4_3_weights.nnd")});
    auto conv4_3_b = file::create({ engine, join_path(weights_dir, "conv4_3_bias.nnd") });
    auto conv4_3 = convolution("conv4_3",
        conv4_2,
        { conv4_3_w },
        { conv4_3_b },
        { 1, 1, 1, 1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto pool4 = pooling("pool4",
        conv4_3,
        pooling_mode::max,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv5_1_w = file::create({ engine, join_path(weights_dir, "conv5_1_weights.nnd")});
    auto conv5_1_b = file::create({ engine, join_path(weights_dir, "conv5_1_bias.nnd") });
    auto conv5_1 = convolution("conv5_1",
        pool4,
        { conv5_1_w },
        { conv5_1_b },
        { 1,1,1,1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto conv5_2_w = file::create({ engine, join_path(weights_dir, "conv5_2_weights.nnd")});
    auto conv5_2_b = file::create({ engine, join_path(weights_dir, "conv5_2_bias.nnd") });
    auto conv5_2 = convolution("conv5_2",
        conv5_1,
        { conv5_2_w },
        { conv5_2_b },
        { 1,1,1,1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto conv5_3_w = file::create({ engine, join_path(weights_dir, "conv5_3_weights.nnd")});
    auto conv5_3_b = file::create({ engine, join_path(weights_dir, "conv5_3_bias.nnd") });
    auto conv5_3 = convolution("conv5_3",
        conv5_2,
        { conv5_3_w },
        { conv5_3_b },
        { 1,1,1,1 },
        { 0, 0, -1, -1 },
        { 1, 1, 1, 1 },
        true); // negative slope for RELU

    auto pool5 = pooling("pool5",
        conv5_3,
        pooling_mode::max,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto fc6_w = file::create({ engine, join_path(weights_dir, "fc6_weights.nnd")});
    auto fc6_b = file::create({ engine, join_path(weights_dir, "fc6_bias.nnd") });
    auto fc6 = fully_connected("fc6",
        pool5,
        fc6_w,
        fc6_b,
        true,
        0
    );

    auto fc7_w = file::create({ engine, join_path(weights_dir, "fc7_weights.nnd")});
    auto fc7_b = file::create({ engine, join_path(weights_dir, "fc7_bias.nnd") });
    auto fc7 = fully_connected("fc7",
        fc6,
        fc7_w,
        fc7_b,
        true,
        0
    );

    auto fc8_w = file::create({ engine, join_path(weights_dir, "fc8_weights.nnd")});
    auto fc8_b = file::create({ engine, join_path(weights_dir, "fc8_bias.nnd") });
    auto fc8 = fully_connected("fc8",
        fc7,
        fc8_w,
        fc8_b,
        true,
        0
    );

    auto softmax = cldnn::softmax(
        "output",
        fc8);

    cldnn::topology topology{
        input,
        reordered_input, reorder_mean,
        conv1_1, conv1_1_w, conv1_1_b,
        conv1_2, conv1_2_w, conv1_2_b,
        pool1 };
    topology.add(
        conv2_1, conv2_1_w, conv2_1_b,
        conv2_2, conv2_2_w, conv2_2_b,
        pool2);
    topology.add(
        conv3_1, conv3_1_w, conv3_1_b,
        conv3_2, conv3_2_w, conv3_2_b,
        conv3_3, conv3_3_w, conv3_3_b,
        pool3);
    topology.add(
        conv4_1, conv4_1_w, conv4_1_b,
        conv4_2, conv4_2_w, conv4_2_b,
        conv4_3, conv4_3_w, conv4_3_b,
        pool4);
    topology.add(
        conv5_1, conv5_1_w, conv5_1_b,
        conv5_2, conv5_2_w, conv5_2_b,
        conv5_3, conv5_3_w, conv5_3_b,
        pool5);
    topology.add(
        fc6, fc6_w, fc6_b,
        fc7, fc7_w, fc7_b,
        fc8, fc8_w, fc8_b,
        softmax);

    return topology;
}

static primitive_id add_conv_layer(const std::string& weights_dir, const engine& engine, topology& topology_inst,
    const std::string& layer_name, const primitive_id& input, const layout weights_layout, const layout bias_layout, const bool use_existing_weights,
    const tensor& padding = { 0, 0, 0, 0 }, const tensor& stride = { 1, 1, 1, 1 }, const bool add_relu = true)
{
    auto weights_data = file::create_mutable({ engine, join_path(weights_dir, layer_name + "_weights.nnd") }, use_existing_weights ? false : true, weights_layout, cldnn::mutable_data::filler_type::xavier);
    auto bias_data = file::create_mutable({ engine, join_path(weights_dir, layer_name + "_bias.nnd") }, use_existing_weights ? false : true, bias_layout, cldnn::mutable_data::filler_type::zero);

    auto conv_layer = convolution(
        layer_name,
        input,
        { weights_data },
        { bias_data },
        stride,
        padding,
        { 1, 1, 1, 1 },
        add_relu);

    topology_inst.add(weights_data, bias_data, conv_layer);

    return conv_layer;
}

static primitive_id add_fc_layer(const std::string& weights_dir, const engine& engine, topology& topology_inst,
    const std::string& layer_name, const primitive_id& input, const layout weights_layout, const layout bias_layout, const bool use_existing_weights)
{
    auto weights_data = file::create_mutable({ engine, join_path(weights_dir, layer_name + "_weights.nnd") }, use_existing_weights ? false : true, weights_layout, cldnn::mutable_data::filler_type::xavier);
    auto bias_data = file::create_mutable({ engine, join_path(weights_dir, layer_name + "_bias.nnd") }, use_existing_weights ? false : true, bias_layout, cldnn::mutable_data::filler_type::zero);

    auto fc_layer = fully_connected(
        layer_name,
        input,
        weights_data,
        bias_data,
        true);

    topology_inst.add(weights_data, bias_data, fc_layer);

    return fc_layer;
}

cldnn::topology build_vgg16_train(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size, bool use_existing_weights)
{
    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { batch_size, 3, 224, 224 };
    auto input = cldnn::input_layout("input", input_layout);
    auto labels = cldnn::input_layout("labels", { input_layout.data_type, format::bfyx,{ batch_size, 1, 1, 1 } });

    topology topology_inst{ input, labels };

    // subtract mean values
    auto reorder_mean = file::create({ engine, join_path(weights_dir, "imagenet_mean.nnd") });
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, cldnn::format::bfyx, input_layout.size },
        reorder_mean);


    auto conv1_1_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 64, 3, 3, 3 } };
    auto conv1_1_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 64, 1 } };
    auto conv1_1 = add_conv_layer(weights_dir, engine, topology_inst, "conv1_1", reordered_input,  conv1_1_w_mem_layout, conv1_1_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto conv1_2_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 64, 64, 3, 3 } };
    auto conv1_2_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 64, 1 } };
    auto conv1_2 = add_conv_layer(weights_dir, engine, topology_inst, "conv1_2", conv1_1, conv1_2_w_mem_layout, conv1_2_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto pool1_argmax_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_size, 64, 112, 112 } });
    auto pool1_argmax = mutable_data("pool1_argmax", pool1_argmax_mem);

    auto pool1 = pooling("pool1",
        conv1_2,
        pool1_argmax,
        pooling_mode::max_with_argmax,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv2_1_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 128, 64, 3, 3 } };
    auto conv2_1_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 128, 1 } };
    auto conv2_1 = add_conv_layer(weights_dir, engine, topology_inst, "conv2_1", pool1, conv2_1_w_mem_layout, conv2_1_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto conv2_2_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 128, 128, 3, 3 } };
    auto conv2_2_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 128, 1 } };
    auto conv2_2 = add_conv_layer(weights_dir, engine, topology_inst, "conv2_2", conv2_1, conv2_2_w_mem_layout, conv2_2_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto pool2_argmax_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_size, 128, 56, 56 } });
    auto pool2_argmax = mutable_data("pool2_argmax", pool2_argmax_mem);

    auto pool2 = pooling("pool2",
        conv2_2,
        pool2_argmax,
        pooling_mode::max_with_argmax,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv3_1_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 256, 128, 3, 3 } };
    auto conv3_1_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 256, 1 } };
    auto conv3_1 = add_conv_layer(weights_dir, engine, topology_inst, "conv3_1", pool2, conv3_1_w_mem_layout, conv3_1_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto conv3_2_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 256, 256, 3, 3 } };
    auto conv3_2_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 256, 1 } };
    auto conv3_2 = add_conv_layer(weights_dir, engine, topology_inst, "conv3_2", conv3_1, conv3_2_w_mem_layout, conv3_2_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto conv3_3_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 256, 256, 3, 3 } };
    auto conv3_3_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 256, 1 } };
    auto conv3_3 = add_conv_layer(weights_dir, engine, topology_inst, "conv3_3", conv3_2, conv3_3_w_mem_layout, conv3_3_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto pool3_argmax_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_size, 256, 28, 28 } });
    auto pool3_argmax = mutable_data("pool3_argmax", pool3_argmax_mem);

    auto pool3 = pooling("pool3",
        conv3_3,
        pool3_argmax,
        pooling_mode::max_with_argmax,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv4_1_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 512, 256, 3, 3 } };
    auto conv4_1_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 512, 1 } };
    auto conv4_1 = add_conv_layer(weights_dir, engine, topology_inst, "conv4_1", pool3, conv4_1_w_mem_layout, conv4_1_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto conv4_2_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 512, 512, 3, 3 } };
    auto conv4_2_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 512, 1 } };
    auto conv4_2 = add_conv_layer(weights_dir, engine, topology_inst, "conv4_2", conv4_1, conv4_2_w_mem_layout, conv4_2_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto conv4_3_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 512, 512, 3, 3 } };
    auto conv4_3_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 512, 1 } };
    auto conv4_3 = add_conv_layer(weights_dir, engine, topology_inst, "conv4_3", conv4_2, conv4_3_w_mem_layout, conv4_3_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto pool4_argmax_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_size, 512, 14, 14 } });
    auto pool4_argmax = mutable_data("pool4_argmax", pool4_argmax_mem);

    auto pool4 = pooling("pool4",
        conv4_3,
        pool4_argmax,
        pooling_mode::max_with_argmax,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv5_1_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 512, 512, 3, 3 } };
    auto conv5_1_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 512, 1 } };
    auto conv5_1 = add_conv_layer(weights_dir, engine, topology_inst, "conv5_1", pool4, conv5_1_w_mem_layout, conv5_1_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto conv5_2_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 512, 512, 3, 3 } };
    auto conv5_2_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 512, 1 } };
    auto conv5_2 = add_conv_layer(weights_dir, engine, topology_inst, "conv5_2", conv5_1, conv5_2_w_mem_layout, conv5_2_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto conv5_3_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 512, 512, 3, 3 } };
    auto conv5_3_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 512, 1 } };
    auto conv5_3 = add_conv_layer(weights_dir, engine, topology_inst, "conv5_3", conv5_2, conv5_3_w_mem_layout, conv5_3_b_mem_layout,
        use_existing_weights, { 0, 0, -1, -1 });

    auto pool5_argmax_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_size, 512, 7, 7 } });
    auto pool5_argmax = mutable_data("pool5_argmax", pool5_argmax_mem);

    auto pool5 = pooling("pool5",
        conv5_3,
        pool5_argmax,
        pooling_mode::max_with_argmax,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto fc6_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 4096, 512, 7, 7 } };
    auto fc6_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 4096, 1 } };
    auto fc6 = add_fc_layer(weights_dir, engine, topology_inst, "fc6", pool5, fc6_w_mem_layout, fc6_b_mem_layout,
        use_existing_weights);

    auto fc7_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 4096, 1, 1, 1 } };
    auto fc7_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 4096, 1 } };
    auto fc7 = add_fc_layer(weights_dir, engine, topology_inst, "fc7", fc6, fc7_w_mem_layout, fc7_b_mem_layout,
        use_existing_weights);

    auto fc8_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 4096, 1, 1, 1 } };
    auto fc8_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 1000, 1 } };
    auto fc8 = add_fc_layer(weights_dir, engine, topology_inst, "fc8", fc7, fc8_w_mem_layout, fc8_b_mem_layout,
        use_existing_weights);

    auto softmax = cldnn::softmax(
        "softmax",
        fc8);

    topology_inst.add( reordered_input, reorder_mean,
        pool1, pool1_argmax,
        pool2, pool2_argmax,
        pool3, pool3_argmax,
        pool4, pool4_argmax,
        pool5, pool5_argmax,
        softmax);

    return topology_inst;
}