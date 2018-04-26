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
#include <api/CPP/scale.hpp>
#include <api/CPP/mutable_data.hpp>
#include <api/CPP/fully_connected_grad_input.hpp>
#include <api/CPP/fully_connected_grad_weights.hpp>
#include <api/CPP/activation.hpp>
#include <api/CPP/activation_grad.hpp>
#include <api/CPP/max_unpooling.hpp>
#include <api/CPP/convolution_grad_input.hpp>
#include <api/CPP/convolution_grad_weights.hpp>
#include <api/CPP/softmax_loss_grad.hpp>

using namespace cldnn;

// Building lenet network with loading weights & biases from file
cldnn::topology build_lenet(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    // [224x224x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { batch_size, 1, 28, 28 };
    auto input = cldnn::input_layout("input", input_layout);

    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, format::bfyx, input_layout.size });

    auto scale_val = memory::allocate(engine, { input_layout.data_type, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_factor = cldnn::data("scale_factor_val", scale_val);
    auto ptr = scale_val.pointer<float>();
    ptr[0] = 0.00390625f;

    // scale input data
    auto scale_input = scale(
        "scale_input",
        reordered_input,
        scale_factor
        );

    auto conv1_w = file::create({ engine, join_path(weights_dir, "conv1_weights.nnd")});
    auto conv1_b = file::create({ engine, join_path(weights_dir, "conv1_bias.nnd")});
    auto conv1 = convolution("conv1",
        scale_input,
        { conv1_w },
        { conv1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
        { 1, 1, 1, 1 },
        false);

    auto pool1 = pooling("pool1",
        conv1,
        pooling_mode::max,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto conv2_w = file::create({ engine, join_path(weights_dir, "conv2_weights.nnd")});
    auto conv2_b = file::create({ engine, join_path(weights_dir, "conv2_bias.nnd") });
    auto conv2 = convolution("conv2",
        pool1,
        { conv2_w },
        { conv2_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
        { 1, 1, 1, 1 },
        false);

    auto pool2 = pooling("pool2",
        conv2,
        pooling_mode::max,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    auto ip1_w = file::create({ engine, join_path(weights_dir, "ip1_weights.nnd")});
    auto ip1_b = file::create({ engine, join_path(weights_dir, "ip1_bias.nnd") });
    auto ip1 = fully_connected("ip1",
        pool2,
        ip1_w,
        ip1_b,
        true,
        0
    );

    auto ip2_w = file::create({ engine, join_path(weights_dir, "ip2_weights.nnd")});
    auto ip2_b = file::create({ engine, join_path(weights_dir, "ip2_bias.nnd") });
    auto ip2 = fully_connected("ip2",
        ip1,
        ip2_w,
        ip2_b,
        false
    );

    auto softmax = cldnn::softmax(
        "output",
        ip2);

    return topology(
        input, reordered_input,
        scale_factor, scale_input,
        conv1, conv1_w, conv1_b, pool1,
        conv2, conv2_w, conv2_b, pool2,
        ip1, ip1_w, ip1_b,
        ip2, ip2_w, ip2_b,
        softmax
        );
}


// Building lenet network for training
cldnn::topology build_lenet_train(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    input_layout.size = { batch_size, 1, 28, 28 };
    auto input = cldnn::input_layout("input", input_layout);
    auto labels = cldnn::input_layout("labels", { input_layout.data_type, format::bfyx,{ batch_size, 1, 1, 1 } });

    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, format::bfyx, input_layout.size });

    auto scale_val = memory::allocate(engine, { input_layout.data_type, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_factor = cldnn::data("scale_factor_val", scale_val);
    auto ptr = scale_val.pointer<float>();
    ptr[0] = 0.00390625f;

    // scale input data
    auto scale_input = scale(
        "scale_input",
        reordered_input,
        scale_factor
    );

    //auto conv1_w_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 20, 1, 5, 5 } });
    //auto conv1_b_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 20, 1 } });
    //auto conv1_w = mutable_data("conv1_weights", conv1_w_mem);
    //auto conv1_b = mutable_data("conv1_bias", conv1_b_mem);
    auto conv1_w = file::create_mutable({ engine, join_path(weights_dir, "conv1_weights.nnd") });
    auto conv1_b = file::create_mutable({ engine, join_path(weights_dir, "conv1_bias.nnd") });

    auto conv1 = convolution("conv1",
        scale_input,
        { conv1_w },
        { conv1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
        { 1, 1, 1, 1 },
        false);

    auto pool1_argmax_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_size, 20, 12, 12 } });
    auto pool1_argmax = mutable_data("pool1_argmax", pool1_argmax_mem);

    auto pool1 = pooling("pool1",
        conv1,
        pool1_argmax,
        pooling_mode::max_with_argmax,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    //auto conv2_w_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 50, 20, 5, 5 } });
    //auto conv2_b_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 50, 1 } });
    //auto conv2_w = mutable_data("conv2_weights", conv2_w_mem);
    //auto conv2_b = mutable_data("conv2_bias", conv2_b_mem);
    auto conv2_w = file::create_mutable({ engine, join_path(weights_dir, "conv2_weights.nnd") });
    auto conv2_b = file::create_mutable({ engine, join_path(weights_dir, "conv2_bias.nnd") });

    auto conv2 = convolution("conv2",
        pool1,
        { conv2_w },
        { conv2_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
        { 1, 1, 1, 1 },
        false);

    auto pool2_argmax_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { batch_size, 50, 4, 4 } });
    auto pool2_argmax = mutable_data("pool2_argmax", pool2_argmax_mem);

    auto pool2 = pooling("pool2",
        conv2,
        pool2_argmax,
        pooling_mode::max_with_argmax,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );

    //auto ip1_w_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 500, 50, 4, 4 } });
    //auto ip1_b_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 500, 1 } });
    //auto ip1_w = mutable_data("ip1_weights", ip1_w_mem);
    //auto ip1_b = mutable_data("ip1_bias", ip1_b_mem);
    auto ip1_w = file::create_mutable({ engine, join_path(weights_dir, "ip1_weights.nnd") });
    auto ip1_b = file::create_mutable({ engine, join_path(weights_dir, "ip1_bias.nnd") });

    auto ip1 = fully_connected("ip1",
        pool2,
        ip1_w,
        ip1_b,
        false
    );

    auto ip1_relu = activation("ip1_relu",
        ip1,
        activation_relu
    );

    //auto ip2_w_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 10, 1, 500, 1 } });
    //auto ip2_b_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 10, 1 } });
    //auto ip2_w = mutable_data("ip2_weights", ip2_w_mem);
    //auto ip2_b = mutable_data("ip2_bias", ip2_b_mem);
    auto ip2_w = file::create_mutable({ engine, join_path(weights_dir, "ip2_weights.nnd") });
    auto ip2_b = file::create_mutable({ engine, join_path(weights_dir, "ip2_bias.nnd") });

    auto ip2 = fully_connected("ip2",
        ip1_relu,
        ip2_w,
        ip2_b,
        false
    );

    auto softmax = cldnn::softmax(
        "softmax",
        ip2);

    auto softmax_loss_grad = cldnn::softmax_loss_grad(
        "softmax_loss_grad",
        softmax,
        labels);

    auto ip2_grad_input = fully_connected_grad_input("ip2_grad_input",
        softmax_loss_grad,
        ip1_relu,
        ip2_w
    );

    auto ip2_grad_weights = fully_connected_grad_weights("ip2_grad_weights",
        softmax_loss_grad,
        ip1_relu,
        ip2_w,
        ip2_b
    );

    auto ip1_relu_grad = activation_grad("ip1_relu_grad",
        ip2_grad_input,
        ip1,
        activation_grad_relu
    );

    auto ip1_grad_input = fully_connected_grad_input("ip1_grad_input",
        ip1_relu_grad,
        pool2,
        ip1_w
    );

    auto ip1_grad_weights = fully_connected_grad_weights("ip1_grad_weights",
        ip1_relu_grad,
        pool2,
        ip1_w,
        ip1_b
    );

    auto pool2_grad = max_unpooling("pool2_grad",
        ip1_grad_input,
        pool2_argmax,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );
    
    auto conv2_grad_input = convolution_grad_input("conv2_grad_input",
        pool2_grad,
        { conv2_w },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 });
    
    auto conv2_grad_weights = convolution_grad_weights("conv2_grad_weights",
        pool2_grad,
        pool1,
        { conv2_w },
        { conv2_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
        { 1, 1, 1, 1 });
    
    auto pool1_grad = max_unpooling("pool1_grad",
        conv2_grad_input,
        pool1_argmax,
        { 1,1,2,2 }, // kernel
        { 1,1,2,2 } // strd
    );
    
    auto conv1_grad_weights = convolution_grad_weights("output",
        pool1_grad,
        scale_input,
        { conv1_w },
        { conv1_b },
        { 1, 1, 1, 1 },
        { 0, 0, 0, 0 },
        { 1, 1, 1, 1 });

    return topology(
        input, reordered_input,
        scale_factor, scale_input,
        conv1, conv1_w, conv1_b,
        pool1_argmax, pool1,
        conv2, conv2_w, conv2_b,
        pool2_argmax, pool2,
        ip1, ip1_w, ip1_b,
        ip1_relu,
        ip2, ip2_w, ip2_b,
        softmax,
        //backward pass
        softmax_loss_grad, labels,
        ip2_grad_input, ip2_grad_weights,
        ip1_relu_grad,
        ip1_grad_input, ip1_grad_weights,
        pool2_grad,
        conv2_grad_input, conv2_grad_weights,
        pool1_grad,
        conv1_grad_weights
    );
}