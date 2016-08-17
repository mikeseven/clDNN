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

#include "implementation_map.h"

namespace neural {

relu::arguments::arguments(neural::engine::type engine, memory::format::type out_fmt, primitive in)
    : engine(engine)
    , negative_slope(0.0f) 
{
    if (in.id() == type_id<const memory>()->id)
    {
        input = { in };
    }
    else
    {
        input = { in.output[0] };
    }

    input_offset =
    {
        input[0].primitive().as<const memory&>().argument.size.batch.size(),
        input[0].primitive().as<const memory&>().argument.size.spatial.size(),
        input[0].primitive().as<const memory&>().argument.size.feature.size(),
    };
    output_size = input[0].primitive().as<const memory&>().argument.size;
    output = { memory::allocate({engine, out_fmt, output_size}) };
    output_offset = 
    {
        output_size.batch.size(),
        output_size.spatial.size(),
        output_size.feature.size(),
    };
}

relu::arguments::arguments(neural::engine::type engine, primitive out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, float slp)
    : engine(engine)
    , output({ out })
    , output_offset({ out_off })
    , output_size({ out_siz })
    , input({ in })
    , input_offset({ in_off })
    , negative_slope(slp) {}

relu::arguments::arguments(neural::engine::type engine, primitive out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off)
    : engine(engine)
    , output({ out })
    , output_offset({ out_off })
    , output_size({ out_siz })
    , input({ in })
    , input_offset({ in_off })
    , negative_slope(0.0f) {}

relu::arguments::arguments(neural::engine::type engine, primitive out, primitive in, float slp)
    : engine(engine)
    , output({ out })
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input({ in })
    , input_offset(in.as<const memory&>().argument.size.batch.size(), in.as<const memory&>().argument.size.spatial.size(), in.as<const memory&>().argument.size.feature.size())
    , negative_slope(slp) {}

relu::arguments::arguments(neural::engine::type engine, primitive out, primitive in)
    : engine(engine)
    , output({ out })
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , negative_slope(0.0f) 
{
    auto input_mem = in.id() == type_id<const memory>()->id ? in : in.output[0];
    input = { input_mem };
    input_offset = {
        input_mem.as<const memory&>().argument.size.batch.size(),
        input_mem.as<const memory&>().argument.size.spatial.size(),
        input_mem.as<const memory&>().argument.size.feature.size()
    };
}

relu_backward::arguments::arguments(neural::engine::type engine, primitive out, neural::vector<uint32_t> out_offset, neural::vector<uint32_t> out_size, std::vector<primitive_at> in, std::vector<neural::vector<uint32_t>> in_offsets, float neg_slope)
    : engine(engine)
    , output({ out })
    , output_offset(out_offset)
    , output_size(out_size)
    , input(in)
    , input_offset(in_offsets)
    , negative_slope(neg_slope) {}

relu_backward::arguments::arguments(neural::engine::type engine, primitive out, std::vector<primitive_at> in, float neg_slope)
    : engine(engine)
    , output({ out })
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input(in)
    , input_offset({ in.size(),{ in[0].primitive().as<const memory&>().argument.size.batch.size(), in[0].primitive().as<const memory&>().argument.size.spatial.size(), in[0].primitive().as<const memory&>().argument.size.feature.size() } })
    , negative_slope(neg_slope) {}

// creates primitive with relu implementation that supports provided arguments
primitive relu::create(relu::arguments arg) {
    auto& input_offset = arg.input_offset;
    auto& output_offset = arg.output_offset;
    auto& output_size = arg.output_size;

    auto& input_arg  = arg.input[0].primitive().as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;

    if (input_arg.size.raw.size() != output_arg.size.raw.size())    throw std::runtime_error("ReLU input/output number of dimension does not match.");
    for (auto &x : input_offset.raw)  if (x < 0)                    throw std::runtime_error("ReLU negative input offset.");

    for (size_t i = 0; i < input_arg.size.raw.size(); ++i) {
        if (input_arg.size.raw[i]  < output_size.raw[i] + input_offset.raw[i]) throw std::runtime_error("ReLU input/output size does not match.");
        if (output_arg.size.raw[i] < output_size.raw[i] + output_offset.raw[i]) throw std::runtime_error("ReLU sizes to small.");
    }

    return is_a_primitive::create<relu>(arg);
}


primitive relu_backward::create(relu_backward::arguments arg) {
    if (arg.input.size() != 2)
        throw std::runtime_error("ReLU backward: number of inputs is incorrect.");

    if (arg.output.size() != 1)
        throw std::runtime_error("ReLU backward: number of outputs is incorrect.");

    auto& forward_output_grad_arg    = arg.input[0].primitive().as<const memory&>().argument;
    auto& forward_output_grad_offset = arg.input_offset[0];

    auto& forward_input_arg    = arg.input[1].primitive().as<const memory&>().argument;
    auto& forward_input_offset = arg.input_offset[1];

    auto& forward_input_grad_arg    = arg.output[0].as<const memory&>().argument;
    auto& forward_input_grad_offset = arg.output_offset;

    auto& processed_window_sizes = arg.output_size;
    if (forward_output_grad_arg.size.raw.size() != forward_input_arg.size.raw.size() || forward_input_arg.size.raw.size() != forward_input_grad_arg.size.raw.size())
        throw std::runtime_error("ReLU backward: number of IO dimension does not match.");

    if (forward_output_grad_arg.format != forward_input_arg.format || forward_input_arg.format != forward_input_grad_arg.format)
        throw std::runtime_error("ReLU backward: IO data format does not match.");

    for (size_t i = 0; i < forward_output_grad_arg.size.raw.size(); ++i) {
        if (forward_output_grad_arg.size.raw[i] < processed_window_sizes.raw[i] + forward_output_grad_offset.raw[i])    throw std::runtime_error("ReLU backward: forward_output_grad size does not match the offset.");
        if (forward_input_arg.size.raw[i]       < processed_window_sizes.raw[i] + forward_input_offset.raw[i])          throw std::runtime_error("ReLU backward: forward_input size does not match the offset.");
        if (forward_input_grad_arg.size.raw[i]  < processed_window_sizes.raw[i] + forward_input_grad_offset.raw[i])     throw std::runtime_error("ReLU backward: forward_input_grad size does not match the offset.");
    }

    return is_a_primitive::create<relu_backward>(arg);
}
}