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
, input({in, mean})
{
};

// creates primitive with fully_connected implementation that supports provided arguments
primitive mean_subtract::create(mean_subtract::arguments arg) {
    auto& input_arg = arg.input[0].primitive.as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;
    auto& mean_arg = arg.input[1].primitive.as<const memory&>().argument;

    if (input_arg.format != memory::format::yxfb_f32)
    {
        throw std::runtime_error("Mean subtract input is not in yxfb format!");
    }
    if (output_arg.format != memory::format::yxfb_f32)
    {
        throw std::runtime_error("Mean subtract output is not in yxfb format!");
    }
    if (mean_arg.format != memory::format::yxfb_f32)
    {
        throw std::runtime_error("Mean subtract mean is not in yxfb format!");
    }

    return is_a_primitive::create<mean_subtract>(arg);
}

}