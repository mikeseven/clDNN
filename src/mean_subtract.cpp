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

mean_subtract::arguments::arguments( neural::engine::type   eng,
                                       primitive            out,
                                       primitive            in,
                                       primitive            mean)
: engine(eng)
, output({out})
, input({in, mean}) {}

mean_subtract::arguments::arguments(neural::engine::type            eng,
                                       neural::memory::format::type out_fmt,
                                       primitive                    in,
                                       primitive                    mean)
: engine(eng)
{
    // if input is previouse layer, not memory primitive need to set input to output memory of this primitive
    auto input_mem = in.id() == type_id<const memory>()->id ? in : in.output[0];
    if (in.id() != type_id<const memory>()->id) {
        input = { in.output[0], mean };
    }
    else {
        input = { in, mean };
    }

    auto input_arg = input_mem.as<const memory&>().argument;
    neural::vector<uint32_t> output_size = {
        input_arg.size.batch[0],
        { { input_arg.size.spatial[0], input_arg.size.spatial[1] } },
        input_arg.size.feature[0]
    };

    output = { memory::allocate({ eng, out_fmt, output_size }) };
}

// creates primitive with fully_connected implementation that supports provided arguments
primitive mean_subtract::create(mean_subtract::arguments arg) {
    auto& input_arg = arg.input[0].primitive().as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;
    auto& mean_arg = arg.input[1].primitive().as<const memory&>().argument;

    if (input_arg.format != memory::format::yxfb_f32)
    {
        throw std::runtime_error("Mean subtract input is not in yxfb format!");
    }
    if (output_arg.format != memory::format::yxfb_f32)
    {
        throw std::runtime_error("Mean subtract output is not in yxfb format!");
    }
    if (mean_arg.format != memory::format::yxfb_f32 && 
        mean_arg.format != memory::format::bfyx_f32)
    {
        throw std::runtime_error("Mean subtract mean is not in yxfb or bfyx format!");
    }
    
    return is_a_primitive::create<mean_subtract>(arg);
}

}