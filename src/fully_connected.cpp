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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "api/neural.h"
#include "multidimensional_counter.h"
#include "implementation_map.h"

namespace neural {

fully_connected::arguments::arguments(
    primitive            out,
    primitive            in,
    primitive            weights,
    primitive            bias,
    bool                 use_relu,
    float                negative_slope)
    : output({ out })
    , input({ in, weights, bias })
    , use_relu(use_relu)
    , negative_slope(negative_slope) {}

fully_connected::arguments::arguments(
    neural::memory::format::type out_fmt,
    primitive                    in,
    primitive                    weights,
    primitive                    bias,
    bool                         use_relu,
    float                        negative_slope)
    : use_relu(use_relu)
    , negative_slope(negative_slope)
{
    // if input is previouse layer, not memory primitive need to set input to output memory of this primitive
    const auto& input_mem = get_memory_primitive(in);
    if (in.id() != type_id<const memory>()->id) {
        input = { in.output[0], weights, bias };
    }
    else {
        input = { in, weights, bias };
    }

    neural::vector<uint32_t> output_size = {
        input_mem.argument.size.batch[0],
        { { get_memory_primitive(bias).argument.size.spatial[0] } },
        1
    };

    output = { memory::allocate({ out_fmt, output_size }) };
}

// creates primitive with fully_connected implementation that supports provided arguments
primitive fully_connected::create(fully_connected::arguments arg) {
    auto& input_arg = arg.input[0].primitive().as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;
    
    if (input_arg.format == memory::format::yxfb_f32 ||
        input_arg.format == memory::format::yxfb_f16)
    {
        // NOTE: Testing for supported weights format is now inside each device implementation of the primitve (e.g. fully_connected_gpu).
    }
    else
    {
        if (input_arg.size.raw.size() != output_arg.size.raw.size())
            throw std::runtime_error("Fully connected input/output number of dimension does not match.");
    }

    return is_a_primitive::create<fully_connected>(arg);
}

}