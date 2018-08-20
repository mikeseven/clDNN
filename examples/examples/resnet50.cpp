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
#include <api/CPP/batch_norm.hpp>
#include <api/CPP/scale.hpp>
#include <api/CPP/mutable_data.hpp>
#include <api/CPP/activation.hpp>
#include <api/CPP/softmax_loss_grad.hpp>
#include <api/CPP/fully_connected_grad_input.hpp>
#include <api/CPP/fully_connected_grad_weights.hpp>
#include <api/CPP/activation_grad.hpp>
#include <api/CPP/max_unpooling.hpp>
#include <api/CPP/average_unpooling.hpp>
#include <api/CPP/convolution_grad_input.hpp>
#include <api/CPP/convolution_grad_weights.hpp>
#include <api/CPP/scale_grad_input.hpp>
#include <api/CPP/scale_grad_weights.hpp>
#include <api/CPP/batch_norm_grad.hpp>

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

static primitive_id add_conv_layer_train(const string& weights_dir, const engine& engine, topology& topology_inst,
                                         const string& layer_name, const primitive_id& input,
                                         const int feature_in, const int feature_out, const int kernel_size,
                                         const bool use_existing_weights, const tensor& padding = { 0, 0, 0, 0 }, 
                                         const tensor& stride = { 1, 1, 1, 1 }, const bool add_relu = false)
{
    auto conv_w_mem_layout = layout{ data_types::f32, format::bfyx, { feature_out, feature_in, kernel_size, kernel_size } };
    auto weights_data = file::create_mutable({ engine, join_path(weights_dir, layer_name + "_weights.nnd") }, use_existing_weights ? false : true, conv_w_mem_layout, cldnn::mutable_data::filler_type::xavier);
    
    auto conv_layer = convolution(
        layer_name,
        input,
        { weights_data },
        stride,
        padding,
        { 1, 1, 1, 1 },
        add_relu);

    topology_inst.add(weights_data, conv_layer);

    return conv_layer;
}

static primitive_id add_conv_layer_grad(const string& weights_dir, const engine& engine, topology& topology_inst,
    const string& layer_name, const primitive_id& input, const primitive_id& input_grad,
    const int feature_in, const int feature_out, const int kernel_size,
    const bool use_existing_weights, const tensor& padding = { 0, 0, 0, 0 },
    const tensor& stride = { 1, 1, 1, 1 }, const bool add_relu = false)
{
    auto conv_input = convolution_grad_input(
        layer_name + "_prev_input",
        input_grad,
        { layer_name + "_weights.nnd" },
        stride,
        padding
    );

    auto conv_weights = convolution_grad_weights(
        layer_name + "_prev_weights",
        input_grad,
        input,
        { layer_name + "_weights.nnd" },
        stride,
        padding,
        { 1,1,1,1 },
        conv_input
    );

    topology_inst.add(conv_input, conv_weights);

    return conv_input;
}

static primitive_id add_batch_norm_layer(const string& weights_dir, const engine& engine, topology& topology_inst, 
										 const string& layer_name, const primitive_id& input, const int featur_size,
										 const bool use_existing_weights)
{
    auto inv_variance_mem_layout = layout{ data_types::f32, format::bfyx, {1, featur_size, 1, 1 } };
    auto inv_variance = file::create_mutable({ engine, join_path(weights_dir, layer_name + "_var.nnd") }, use_existing_weights ? false : true, inv_variance_mem_layout, cldnn::mutable_data::filler_type::zero);

    float epsilon = 0.0001f;
    auto batch_norm_layer = batch_norm(
        layer_name,
        input,
        epsilon,
		inv_variance);

    topology_inst.add(batch_norm_layer, inv_variance);

    return batch_norm_layer;
}

static primitive_id add_batch_norm_layer_grad(topology& topology_inst,
    const string& layer_name, const primitive_id& input, const primitive_id& input_grad)
{
    auto batch_norm_grad_layer = batch_norm_grad(
        layer_name + "_prev",
        input_grad,
        input,
        layer_name + "_var.nnd"
    );

    topology_inst.add(batch_norm_grad_layer);

    return batch_norm_grad_layer;
}

static primitive_id add_scale_layer(const string& weights_dir, const engine& engine, topology& topology_inst, 
                                    const string& layer_name, const primitive_id& input, 
                                    const int feature_size, const bool use_existing_weights)
{
    auto scale_mem_layout = layout{ data_types::f32, format::bfyx, {1, feature_size, 1, 1} };
    auto scale_input = file::create_mutable({ engine, join_path(weights_dir, layer_name + "_scale_input.nnd") }, use_existing_weights ? false : true, scale_mem_layout, cldnn::mutable_data::filler_type::xavier);
	auto scale_bias = file::create_mutable({ engine, join_path(weights_dir, layer_name + "_bias.nnd") }, use_existing_weights ? false : true, scale_mem_layout, cldnn::mutable_data::filler_type::xavier);

	auto mul_layer = cldnn::scale(
		layer_name,
		input,
		scale_input,
        scale_bias
	);
	topology_inst.add(scale_input, scale_bias, mul_layer);

	return mul_layer;
}

static primitive_id add_scale_layer_grad(topology& topology_inst, const string& layer_name, const primitive_id& input, const primitive_id& input_grad)
{
    auto scale_input = scale_grad_input(
        layer_name + "_prev_input",
        input,
        layer_name + "_scale_input.nnd"
    );

    auto scale_weights = scale_grad_weights(
        layer_name + "_prev_weights",
        input,
        input_grad,
        layer_name + "_scale_input.nnd",
        layer_name + "_bias.nnd",
        scale_input,
        cldnn::padding()
    );

    topology_inst.add(scale_input, scale_weights);

    return scale_input;
}

static primitive_id add_activation_layer(topology& topology_inst, const string& layer_name, const primitive_id& input)
{
    auto activ = activation(
        layer_name, 
        input, 
        activation_relu);

    topology_inst.add(activ);

    return activ;
}

static primitive_id add_activation_grad_layer(topology& topology_inst, const string& layer_name, const primitive_id& input, const primitive_id& input_grad)
{
    auto activ = activation_grad(
        layer_name,
        input_grad,
        input,
        activation_grad_relu
    );

    topology_inst.add(activ);

    return activ;
}

static primitive_id add_block(const string& weights_dir, const engine& engine, topology& topology_inst,
                              const string& block_name, const primitive_id& input,
                              const int feature_in, const int feature_out, 
							  const int kernel_size, const bool use_existing_weights, bool with_activation, 
                              const tensor& padding = { 0, 0, 0, 0 }, const tensor& res_stride = { 1, 1, 1, 1 })
{
    auto conv = add_conv_layer_train(weights_dir, engine, topology_inst, "res" + block_name, input, feature_in,
                                     feature_out, kernel_size, use_existing_weights, padding, res_stride);
    auto batch_morm = add_batch_norm_layer(weights_dir, engine, topology_inst, "bn" + block_name, conv,
                                           feature_out, use_existing_weights);
    auto scale = add_scale_layer(weights_dir, engine, topology_inst, "scale" + block_name, batch_morm,
                                 feature_out, use_existing_weights);

    primitive_id block = scale;
    if (with_activation)
        block = add_activation_layer(topology_inst, "res" + block_name + "_relu", scale);

    return block;
}

static primitive_id add_block_backward(const string& weights_dir, const engine& engine, topology& topology_inst,
    const string& block_name, const primitive_id& input, const primitive_id& conv_relu,
    const int feature_in, const int feature_out,
    const int kernel_size, const bool use_existing_weights, bool with_activation,
    const tensor& padding = { 0, 0, 0, 0 }, const tensor& res_stride = { 1, 1, 1, 1 })
{
    auto block = input;
    if (with_activation)
        block = add_activation_grad_layer(topology_inst, "res" + block_name + "_relu_grad", input, "scale" + block_name);

    auto scale_grad = add_scale_layer_grad(topology_inst, "scale" + block_name, "bn" + block_name,block);
    auto batch_norm_grad = add_batch_norm_layer_grad(topology_inst, "bn" + block_name, "res" + block_name, scale_grad);
    auto conv_grad = add_conv_layer_grad(weights_dir, engine, topology_inst, "res" + block_name, conv_relu, batch_norm_grad, feature_in, feature_out, kernel_size, use_existing_weights, padding, res_stride);

    return conv_grad;
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

static primitive_id add_residual_layers_train(const string& weights_dir, const engine& engine, topology& topology_inst,
                                              const string& res_name, const primitive_id& input, 
                                              const int feature_size, const int feature_in_size, const bool use_existing_weights,
                                              bool conv_in_branch1, const tensor& res_stride = { 1, 1, 1, 1 })
{
    auto branch2a = add_block(weights_dir, engine, topology_inst, res_name + "_branch2a", input, feature_in_size,
                              feature_size, 1, use_existing_weights, true, {0, 0, 0, 0}, res_stride);
    auto branch2b = add_block(weights_dir, engine, topology_inst, res_name + "_branch2b", branch2a, feature_size,
                              feature_size, 3, use_existing_weights, true, {0, 0, -1, -1});
    auto branch2c = add_block(weights_dir, engine, topology_inst, res_name + "_branch2c", branch2b, feature_size,
                              4 * feature_size, 1, use_existing_weights, false);

    primitive_id branch1 = input;
    if (conv_in_branch1)
        branch1 = add_block(weights_dir, engine, topology_inst, res_name + "_branch1", input, feature_in_size,
                            4 * feature_size, 1, use_existing_weights, false, {0, 0, 0, 0}, res_stride);

    auto res_sum = eltwise("res" + res_name, branch1, branch2c, eltwise_mode::sum);
    auto res_relu = add_activation_layer(topology_inst, "res" + res_name + "_relu", res_sum);
    topology_inst.add(res_sum);

    return res_relu;
}

static primitive_id add_residual_layers_backward(const string& weights_dir, const engine& engine, topology& topology_inst,
                                                 const string& res_name, const primitive_id& input, const primitive_id& last_conv_relu,
                                                 const int feature_size, const int feature_in_size, const bool use_existing_weights,
                                                 bool conv_in_branch1, const tensor& res_stride = { 1, 1, 1, 1 })
{
    auto res_relu_grad = add_activation_grad_layer(topology_inst, "res" + res_name + "_relu_grad", "res" + res_name, input);

    primitive_id branch1_grad = res_relu_grad;
    if (conv_in_branch1)
        branch1_grad = add_block_backward(weights_dir, engine, topology_inst, res_name + "_branch1", res_relu_grad, last_conv_relu, feature_in_size, 4 * feature_size, 1, use_existing_weights, false, { 0,0,0,0 }, res_stride);

    auto branch2c_grad = add_block_backward(weights_dir, engine, topology_inst, res_name + "_branch2c", res_relu_grad, "res" + res_name + "_branch2b_relu", feature_size, 4 * feature_size, 1, use_existing_weights, false);
    auto branch2b_grad = add_block_backward(weights_dir, engine, topology_inst, res_name + "_branch2b", branch2c_grad, "res" + res_name + "_branch2a_relu", feature_size, feature_size, 3, use_existing_weights, true, { 0,0,-1,-1 });
    auto branch2a_grad = add_block_backward(weights_dir, engine, topology_inst, res_name + "_branch2a", branch2b_grad, last_conv_relu, feature_in_size, feature_size, 1, use_existing_weights, true, { 0, 0, 0, 0 }, res_stride);

    auto sum = eltwise("res" + res_name + "_prev", branch1_grad, branch2a_grad, eltwise_mode::sum);
    topology_inst.add(sum);

    return sum;
}

// Building resnet50 network with loading weights & biases from file
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


// Building resnet50 network for training
cldnn::topology build_resnet50_train(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size, const bool mean_subtract, bool use_existing_weights, std::vector<cldnn::primitive_id>& outputs)
{
    input_layout.size = { batch_size, 3, 224, 224 };
    auto input = cldnn::input_layout("input", input_layout);
    auto labels = cldnn::input_layout("labels", { input_layout.data_type, format::bfyx,{ batch_size, 1, 1, 1 } });
    topology topology_inst{ input };

    primitive_id corrected_input = input;
    if (mean_subtract)
    {
        // Subtract mean values if necessary.
        auto reorder_mean = file::create({ engine, join_path(weights_dir, "resnet50_mean.nnd") });
        auto reordered_input = reorder(
            "reorder",
            input,
            { input_layout.data_type, format::bfyx, input_layout.size },
            reorder_mean);
        topology_inst.add(reorder_mean, reordered_input);
        corrected_input = reordered_input;
    }
    else
    {
        auto reordered_input = reorder(
            "reorder",
            input,
            { input_layout.data_type, format::bfyx, input_layout.size });
        topology_inst.add(reordered_input);
        corrected_input = reordered_input;
    }

    auto conv1_w_mem_layout = layout{ data_types::f32, format::bfyx,{ 64, 3, 7, 7 } };
    auto conv1_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 64, 1 } };
    auto conv1_w = file::create_mutable({ engine, join_path(weights_dir, "conv1_weights.nnd") }, use_existing_weights ? false : true, conv1_w_mem_layout, cldnn::mutable_data::filler_type::xavier);
    auto conv1_b = file::create_mutable({ engine, join_path(weights_dir, "conv1_bias.nnd") }, use_existing_weights ? false : true, conv1_b_mem_layout, cldnn::mutable_data::filler_type::zero);

    auto conv1 = convolution(
        "conv1",
        corrected_input,
        { conv1_w },
        { conv1_b },
        { 1, 1, 2, 2 },
        { 0, 0, -3, -3 },
        { 1, 1, 1, 1 },
        false
    );
    topology_inst.add(conv1, conv1_w, conv1_b);

    auto bn_conv1 = add_batch_norm_layer(weights_dir, engine, topology_inst, "bn_conv1", conv1, 64, use_existing_weights);
    auto scale_conv1 = add_scale_layer(weights_dir, engine, topology_inst, "scale_conv1", bn_conv1, 64, use_existing_weights);
    auto conv1_relu = add_activation_layer(topology_inst, "con1_relu", scale_conv1);

    auto pool1_argmax_mem = memory::allocate(engine, { data_types::f32, format::bfyx, {batch_size, 64, 56, 56} });
    auto pool1_argmax = mutable_data("pool1_argmax", pool1_argmax_mem);

    auto pool1 = pooling(
        "pool1",
        conv1_relu,
        pool1_argmax,
        pooling_mode::max_with_argmax,
        { 1, 1, 3, 3 },
        { 1, 1, 2, 2 });
    topology_inst.add(pool1, pool1_argmax);

    // [56x56x64xF] res2a->res2b->res2c [56x56x256xF].
    auto res2a = add_residual_layers_train(weights_dir, engine, topology_inst, "2a", pool1, 64, 64, use_existing_weights, true);
    auto res2b = add_residual_layers_train(weights_dir, engine, topology_inst, "2b", res2a, 64, 256, use_existing_weights, false);
    auto res2c = add_residual_layers_train(weights_dir, engine, topology_inst, "2c", res2b, 64, 256, use_existing_weights, false);

    // [56x56x256xF] res3a->res3b->res3c->res3d [28x28x512xF].
    auto res3a = add_residual_layers_train(weights_dir, engine, topology_inst, "3a", res2c, 128, 256, use_existing_weights, true, { 1, 1, 2, 2 });
    auto res3b = add_residual_layers_train(weights_dir, engine, topology_inst, "3b", res3a, 128, 512, use_existing_weights, false);
    auto res3c = add_residual_layers_train(weights_dir, engine, topology_inst, "3c", res3b, 128, 512, use_existing_weights, false);
    auto res3d = add_residual_layers_train(weights_dir, engine, topology_inst, "3d", res3c, 128, 512, use_existing_weights, false);

    // [28x28x512xF] res4a->res4b->res4c->res4d->res4e->res4f [14x14x1024xF].
    auto res4a = add_residual_layers_train(weights_dir, engine, topology_inst, "4a", res3d, 256, 512, use_existing_weights, true, { 1, 1, 2, 2 });
    auto res4b = add_residual_layers_train(weights_dir, engine, topology_inst, "4b", res4a, 256, 1024, use_existing_weights, false);
    auto res4c = add_residual_layers_train(weights_dir, engine, topology_inst, "4c", res4b, 256, 1024, use_existing_weights, false);
    auto res4d = add_residual_layers_train(weights_dir, engine, topology_inst, "4d", res4c, 256, 1024, use_existing_weights, false);
    auto res4e = add_residual_layers_train(weights_dir, engine, topology_inst, "4e", res4d, 256, 1024, use_existing_weights, false);
    auto res4f = add_residual_layers_train(weights_dir, engine, topology_inst, "4f", res4e, 256, 1024, use_existing_weights, false);

    // [14x14x1024xF] res5a->res5b->res5c [7x7x2048xF].
    auto res5a = add_residual_layers_train(weights_dir, engine, topology_inst, "5a", res4f, 512, 1024, use_existing_weights, true, { 1, 1, 2, 2 });
    auto res5b = add_residual_layers_train(weights_dir, engine, topology_inst, "5b", res5a, 512, 2048, use_existing_weights, false);
    auto res5c = add_residual_layers_train(weights_dir, engine, topology_inst, "5c", res5b, 512, 2048, use_existing_weights, false);

    auto pool5 = pooling(
        "pool5",
        res5c,
        pooling_mode::average,
        { 1, 1, 7, 7 },
        { 1, 1, 1, 1 });

    auto fc1000_w_mem_layout = layout{ data_types::f32, format::bfyx, {1000, 2048, 1, 1} };
    auto fc1000_b_mem_layout = layout{ data_types::f32, format::bfyx,{ 1, 1, 1000, 1 } };
    auto fc1000_weights_data = file::create_mutable({ engine, join_path(weights_dir, "fc1000_weights.nnd") }, use_existing_weights ? false : true, fc1000_w_mem_layout, cldnn::mutable_data::filler_type::xavier);
    auto fc1000_bias_data = file::create_mutable({ engine, join_path(weights_dir, "fc1000_bias.nnd") }, use_existing_weights ? false : true, fc1000_w_mem_layout, cldnn::mutable_data::filler_type::zero);

    auto fc1000 = fully_connected(
        "fc1000",
        pool5,
        fc1000_weights_data,
        fc1000_bias_data);

    auto softmax = cldnn::softmax(
        "softmax",
        fc1000);

    topology_inst.add(pool5,
        fc1000_weights_data, fc1000_bias_data, fc1000,
        softmax);

    //backward pass

    auto softmax_loss_grad = cldnn::softmax_loss_grad(
        "softmax_loss_grad",
        softmax,
        labels
    );

    auto fc1000_grad_input = fully_connected_grad_input(
        "fc1000_grad_input",
        softmax_loss_grad,
        pool5,
        fc1000_weights_data
    );

    auto fc1000_weights_data_prev = file::create_mutable({ engine, join_path(weights_dir, "fc1000_weights_prev.nnd") }, use_existing_weights ? false : true, fc1000_w_mem_layout, cldnn::mutable_data::filler_type::zero);
    auto fc1000_bias_data_prev = file::create_mutable({ engine, join_path(weights_dir, "fc1000_bias_prev.nnd") }, use_existing_weights ? false : true, fc1000_w_mem_layout, cldnn::mutable_data::filler_type::zero);

    auto fc1000_grad_weights = fully_connected_grad_weights(
        "fc1000_grad_weights",
        softmax_loss_grad,
        pool5,
        fc1000_weights_data,
        fc1000_bias_data,
        fc1000_weights_data_prev,
        fc1000_bias_data_prev,
        fc1000_grad_input
    );

    auto pool5_grad = average_unpooling(
        "pool5_grad",
        fc1000_grad_input,
        { 1, 2048, 7, 7 },
        { 1, 1, 7, 7 },
        { 1, 1, 1, 1 }
    );

    // [7x7x2048xF] res5c_grad->res5b_grad->res5a_grad [14x14x1024xF].
    auto res5c_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "5c", pool5_grad, "res5b_relu", 512, 2048, use_existing_weights, false);
    auto res5b_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "5b", res5c_grad, "res5a_relu", 512, 2048, use_existing_weights, false);
    auto res5a_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "5a", res5b_grad, "res4f_relu", 512, 1024, use_existing_weights, true, { 1,1,2,2 });

    // [14x14x1024xF] res4f_grad->res4e_grad->res4d_grad->res4c_grad->res4b_grad->res4a_grad [28x28x512xF].
    auto res4f_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "4f", res5a_grad, "res4e_relu", 256, 1024, use_existing_weights, false);
    auto res4e_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "4e", res4f_grad, "res4d_relu", 256, 1024, use_existing_weights, false);
    auto res4d_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "4d", res4e_grad, "res4c_relu", 256, 1024, use_existing_weights, false);
    auto res4c_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "4c", res4d_grad, "res4b_relu", 256, 1024, use_existing_weights, false);
    auto res4b_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "4b", res4c_grad, "res4a_relu", 256, 1024, use_existing_weights, false);
    auto res4a_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "4a", res4b_grad, "res3d_relu", 256, 512, use_existing_weights, true, { 1,1,2,2 });

    // [28x28x512xF] res3d_grad->res3c_grad->res3b_grad->res3a_grad [56x56x256xF].
    auto res3d_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "3d", res4a_grad, "res3c_relu", 128, 512, use_existing_weights, false);
    auto res3c_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "3c", res3d_grad, "res3b_relu", 128, 512, use_existing_weights, false);
    auto res3b_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "3b", res3c_grad, "res3a_relu", 128, 512, use_existing_weights, false);
    auto res3a_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "3a", res3b_grad, "res2c_relu", 128, 256, use_existing_weights, true, { 1,1,2,2 });

    // [56x56x256xF] res2c_grad->res2b_grad->res2a_grad [56x56x64xF].
    auto res2c_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "2c", res3a_grad, "res2b_relu", 64, 256, use_existing_weights, false);
    auto res2b_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "2b", res2c_grad, "res2a_relu", 64, 256, use_existing_weights, false);
    auto res2a_grad = add_residual_layers_backward(weights_dir, engine, topology_inst, "2a", res2b_grad, "pool1", 64, 64, use_existing_weights, true);

    auto pool1_grad = max_unpooling(
        "pool1_grad",
        res2a_grad,
        pool1_argmax,
        { 1, 1, 3, 3 },
        { 1, 1, 2, 2 }
    );

    auto conv1_relu_grad = add_activation_grad_layer(topology_inst, "conv1_relu_grad", pool1_grad, "scale_conv1");
    auto scale_conv1_grad = add_scale_layer_grad(topology_inst, "scale_conv1", conv1_relu_grad, "bn_conv1");
    auto bn_conv1_grad = add_batch_norm_layer_grad(topology_inst, "bn_conv1", scale_conv1_grad, "conv1");

    
    auto conv1_w_prev = file::create_mutable({ engine, join_path(weights_dir, "conv1_weights_prev.nnd") }, use_existing_weights ? false : true, conv1_w_mem_layout, cldnn::mutable_data::filler_type::zero);
    auto conv1_b_prev = file::create_mutable({ engine, join_path(weights_dir, "conv1_bias_prev.nnd") }, use_existing_weights ? false : true, conv1_b_mem_layout, cldnn::mutable_data::filler_type::zero);

    auto conv1_grad_weights = convolution_grad_weights(
        "output",
        bn_conv1_grad,
        corrected_input,
        {"conv1_weights.nnd"},
        {"conv1_bias.nnd"},
        {conv1_w_prev},
        {conv1_b_prev},
        { 1, 1, 2, 2 },
        { 0, 0, -3, -3 },
        { 1, 1, 1, 1 }
    );

    topology_inst.add(softmax_loss_grad, labels, 
                      fc1000_grad_input, 
                      fc1000_grad_weights, fc1000_weights_data_prev, fc1000_bias_data_prev,
                      pool5_grad,
                      pool1_grad,
                      conv1_grad_weights, conv1_w_prev, conv1_b_prev);

    return topology_inst;
}