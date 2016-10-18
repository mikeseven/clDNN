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

#include "api/neural.h"
#include "implementation_map.h"

namespace neural {

    const memory* get_memory_primitive(neural::primitive p)
    {
        return p.id() != type_id<const memory>()->id ? &p.output[0].as<const memory&>() : &p.as<const memory&>();
    }

    depth_concatenate::arguments::arguments(std::vector<primitive_at> in, primitive out)
    : output({out})
    , input({in})
    {}

    depth_concatenate::arguments::arguments(neural::memory::format::type out_fmt, std::vector<primitive_at> in)
        : input({in})
    {
        uint32_t out_depth_count = 0;
        for (auto i : input)
        {
            out_depth_count += i.primitive().as<const memory&>().argument.size.feature[0];
        }
        auto output_size = get_memory_primitive(input[0].primitive())->argument.size;
        output_size.feature[0] = out_depth_count;
        output = { memory::allocate({ out_fmt, output_size }) };
    }

primitive depth_concatenate::create(depth_concatenate::arguments arg) {
    auto& input_arg  = arg.input[0].primitive().as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;

    auto format = input_arg.format;

    uint32_t depth_count = 0;
    auto input_size = input_arg.size;
    for (auto i : arg.input)
    {
        auto& input_mem = i.primitive().as<const memory&>();
        if (input_mem.argument.format != format) throw std::runtime_error("Every input must have the same format!");
        if (input_mem.argument.size.batch[0] != input_size.batch[0]) throw std::runtime_error("Every input must have the same number of batches!");
        if (input_mem.argument.size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Every input must have the same size in X dimension!");
        if (input_size.spatial.size() > 1)
            if (input_mem.argument.size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Every input must have the same size in Y dimension!");
        depth_count += input_mem.argument.size.feature[0];
    }

    if (output_arg.format != format) throw std::runtime_error("Input and output must have the same format!");
    if (depth_count != output_arg.size.feature[0]) throw std::runtime_error("Output depth count mismatch sum of input depths!");
    if (output_arg.size.batch[0] != input_size.batch[0]) throw std::runtime_error("Output batch size must match input batch size!");
    if (output_arg.size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Output X size must match input X size!");
    if (input_size.spatial.size() > 1)
        if (output_arg.size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Output Y size must match input Y size!");

    return is_a_primitive::create<depth_concatenate>(arg);
}

} // namespace neural
