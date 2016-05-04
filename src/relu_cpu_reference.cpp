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

#include "relu_cpu_reference.h"
#include "multidimensional_counter.h"

namespace neural {

relu_cpu_reference::relu_cpu_reference(relu &arg)
    : is_an_implementation(neural::type_id<relu_cpu_reference>())
    , outer(arg)
{};

relu_cpu_reference::~relu_cpu_reference() {}

void relu_cpu_reference::implementation(const void *ptr) {
    auto this_relu = static_cast<const relu *>(ptr);

    auto& input_offset = this_relu->argument.input_offset;
    auto& output_offset = this_relu->argument.output_offset;
    auto& output_size = this_relu->argument.output_size;

    //auto& input_arg  = this_relu->input_memory(0).argument;
    //auto& output_arg = this_relu->output_memory(0).argument;
    auto& input_arg = this_relu->argument.input[0].primitive.as<const memory&>().argument; //todo tmp solution
    auto& output_arg = this_relu->argument.output[0].as<const memory&>().argument;

    if (input_arg.format != memory::format::yxfb_f32)   throw std::runtime_error("ReLU reference uses yxfb_f32 format.");
    if (input_arg.size.raw.size() != output_arg.size.raw.size()) throw std::runtime_error("ReLU input/output number of dimension does not match.");
    if (input_arg.format != output_arg.format)          throw std::runtime_error("ReLU input/output data format does not match.");
    for (auto &x : input_offset.raw)  if (x < 0)                  throw std::runtime_error("ReLU negative input offset.");

    for (size_t i = 0; i < input_arg.size.raw.size(); ++i) {
        if (input_arg.size.raw[i]  < output_size.raw[i] + input_offset.raw[i]) throw std::runtime_error("ReLU input/output size does not match.");
        if (output_arg.size.raw[i] < output_size.raw[i] + output_offset.raw[i]) throw std::runtime_error("ReLU sizes to small.");
    }

    assert(1 == output_size.feature.size());
    assert(1 == output_size.batch.size());

    //auto input  = static_cast<float*>(this_relu->input_memory(0).pointer);
    //auto output = static_cast<float*>(this_relu->output_memory(0).pointer);
    auto input = static_cast<float*>(this_relu->argument.input[0].primitive.as<const memory&>().pointer);  //todo tmp solution
    auto output = static_cast<float*>(this_relu->argument.output[0].as<const memory&>().pointer);

    namespace nd = ndimensional;
    nd::value<uint32_t> range(output_size);
    nd::calculate_idx<uint32_t, memory::format::yxfb_f32> calc_in_idx(input_arg.size);
    nd::calculate_idx<uint32_t, memory::format::yxfb_f32> calc_out_idx(output_arg.size);

    for (auto pos : range) {
        auto in_idx = calc_in_idx(pos + input_offset);
        auto out_idx = calc_out_idx(pos + output_offset);

        output[out_idx] = std::max(input[in_idx], 0.0f) + this_relu->argument.negative_slope * std::min(input[in_idx], 0.0f);
    }
}


relu_backward_cpu_reference::relu_backward_cpu_reference(relu_backward &arg)
: is_an_implementation(neural::type_id<relu_backward_cpu_reference>())
, outer(arg)
{};

relu_backward_cpu_reference::~relu_backward_cpu_reference() {}

void relu_backward_cpu_reference::implementation(const void *ptr)
{
    auto this_relu = static_cast<const relu_backward *>(ptr);

    if (this_relu->input().size() != 2)
        throw std::runtime_error("ReLU backward: number of inputs is incorrect.");

    if (this_relu->output().size() != 1)
        throw std::runtime_error("ReLU backward: number of outputs is incorrect.");

    //auto forward_output_grad = static_cast<float*>(this_relu->input_memory(0).pointer);
    //auto forward_input       = static_cast<float*>(this_relu->input_memory(1).pointer);
    //auto forward_input_grad  = static_cast<float*>(this_relu->output_memory(0).pointer);
    auto forward_output_grad = static_cast<float*>(this_relu->argument.input[0].primitive.as<const memory&>().pointer);
    auto forward_input = static_cast<float*>(this_relu->argument.input[1].primitive.as<const memory&>().pointer);
    auto forward_input_grad = static_cast<float*>(this_relu->argument.output[0].as<const memory&>().pointer);

    //auto forward_output_grad_arg    = this_relu->input_memory(0).argument;
    auto forward_output_grad_arg = this_relu->argument.input[0].primitive.as<const memory&>().argument;
    auto forward_output_grad_offset = this_relu->argument.input_offset[0];

    //auto forward_input_arg    = this_relu->input_memory(1).argument;
    auto forward_input_arg = this_relu->argument.input[1].primitive.as<const memory&>().argument;
    auto forward_input_offset = this_relu->argument.input_offset[1];

    //auto forward_input_grad_arg    = this_relu->output_memory(0).argument;
    auto forward_input_grad_arg = this_relu->argument.output[0].as<const memory&>().argument;
    auto forward_input_grad_offset = this_relu->argument.output_offset;

    auto processed_window_sizes = this_relu->argument.output_size;

    if (forward_output_grad_arg.size.raw.size() != forward_input_arg.size.raw.size() || forward_input_arg.size.raw.size() != forward_input_grad_arg.size.raw.size())
        throw std::runtime_error("ReLU backward: number of IO dimension does not match.");

    if (forward_output_grad_arg.format != forward_input_arg.format || forward_input_arg.format != forward_input_grad_arg.format)
        throw std::runtime_error("ReLU backward: IO data format does not match.");

    for (size_t i = 0; i < forward_output_grad_arg.size.raw.size(); ++i) {
        if (forward_output_grad_arg.size.raw[i] < processed_window_sizes.raw[i] + forward_output_grad_offset.raw[i]) throw std::runtime_error("ReLU backward: forward_output_grad size does not match the offset.");
        if (forward_input_arg.size.raw[i]       < processed_window_sizes.raw[i] + forward_input_offset.raw[i]) throw std::runtime_error("ReLU backward: forward_input size does not match the offset.");
        if (forward_input_grad_arg.size.raw[i]  < processed_window_sizes.raw[i] + forward_input_grad_offset.raw[i]) throw std::runtime_error("ReLU backward: forward_input_grad size does not match the offset.");
    }

    namespace nd = ndimensional;
    nd::value<uint32_t> range(processed_window_sizes);
    nd::calculate_idx<uint32_t, memory::format::yxfb_f32> calc_forward_input_idx(forward_input_arg.size);
    nd::calculate_idx<uint32_t, memory::format::yxfb_f32> calc_forward_output_grad_idx(forward_output_grad_arg.size);
    nd::calculate_idx<uint32_t, memory::format::yxfb_f32> calc_forward_input_grad_idx(forward_input_grad_arg.size);
    for (auto pos : range) {
        auto forward_input_idx = calc_forward_input_idx(pos + forward_input_offset);
        auto forward_output_grad_idx = calc_forward_output_grad_idx(pos + forward_output_grad_offset);
        auto forward_input_grad_idx = calc_forward_input_grad_idx(pos + forward_input_grad_offset);

        forward_input_grad[forward_input_grad_idx] = (forward_input[forward_input_idx] <= 0.0f ? 0.0f : 1.0f) * forward_output_grad[forward_output_grad_idx];
    }
}

namespace {
struct attach {
    attach() {
        auto key_fw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto key_bw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = relu_cpu_reference::create;
        auto val_bw = relu_backward_cpu_reference::create;

        relu_fw_implementation_map::instance().insert( {key_fw, val_fw} ); //todo keys should be different
        relu_bw_implementation_map::instance().insert( {key_bw, val_bw} );
    }
    ~attach() {}
};

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

}
}
