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

fully_connected::arguments::arguments( neural::engine::type eng,
                                       primitive            out,
                                       primitive            in,
                                       primitive            weights,
                                       primitive            bias)
: engine(eng)
, output({out})
, input({in, weights, bias})
{
};

// creates primitive with fully_connected implementation that supports provided arguments
primitive fully_connected::create(fully_connected::arguments arg) {
    auto& input_arg = arg.input[0].primitive.as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;
    auto& weight_arg = arg.input[1].primitive.as<const memory&>().argument;
    
    if (input_arg.format == memory::format::yxfb_f32)
    {
        if(weight_arg.format != memory::format::yxfn_f32)
            throw std::runtime_error("Fully connected input is yxfb, so weights must be in format yxfn!");
    }
    else
    {
        if (input_arg.size.raw.size() != output_arg.size.raw.size())    throw std::runtime_error("Fully connected input/output number of dimension does not match.");
        if (weight_arg.format != memory::format::xb_f32 &&
            weight_arg.format != memory::format::x_f32)                 throw std::runtime_error("Fully connected weight format is not xb_f32 or x_f32 or nb_f32.");
    }

    return is_a_primitive::create<fully_connected>(arg);
}

}