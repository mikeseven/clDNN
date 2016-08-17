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

#include "multidimensional_counter.h"
#include "implementation_map.h"

namespace neural {

normalization::response::arguments::arguments(
                 neural::engine::type aengine,
                 primitive aoutput,
                 primitive ainput,
                 uint32_t  asize,
                 neural::padding::type apadding,
                 float ak,
                 float aalpha,
                 float abeta)
    : engine(aengine)
    , output({ aoutput })
    , output_offset(aoutput.as<const memory&>().argument.size.batch.size(), aoutput.as<const memory&>().argument.size.spatial.size(), aoutput.as<const memory&>().argument.size.feature.size())
    , output_size(aoutput.as<const memory&>().argument.size)
    , input({ ainput })
    , input_offset(ainput.as<const memory&>().argument.size.batch.size(), ainput.as<const memory&>().argument.size.spatial.size(), ainput.as<const memory&>().argument.size.feature.size())
    , size(asize)
    , padding(apadding)
    , k(ak)
    , alpha(aalpha)
    , beta(abeta) {}

normalization::response::arguments::arguments(
                 neural::engine::type aengine,
                 memory::format::type aoutput_fmt,
                 primitive ainput,
                 uint32_t  asize,
                 neural::padding::type apadding,
                 float ak,
                 float aalpha,
                 float abeta)
    : engine(aengine)
    , input({ ainput })
    , size(asize)
    , padding(apadding)
    , k(ak)
    , alpha(aalpha)
    , beta(abeta)
{ 
    if (ainput.output.size() != 1) throw std::runtime_error("should have one output");
    auto input_mem = ainput.output[0].as<const memory&>().argument;
    input_offset =
    {
        input_mem.size.batch.size(),
        input_mem.size.spatial.size(),
        input_mem.size.feature.size()
    };
    output_offset =
    {
        input_mem.size.batch.size(),
        input_mem.size.spatial.size(),
        input_mem.size.feature.size()
    };
    output_size = input_mem.size;
    output = {
        memory::allocate(
        {
            aengine,
            aoutput_fmt,
            output_size
        }) };
}

normalization::response::arguments::arguments(
                 neural::engine::type aengine,
                 primitive aoutput,
                 vector<uint32_t> aoutput_offset,
                 vector<uint32_t> aoutput_size,
                 primitive ainput,
                 vector<int32_t> ainput_offset,
                 uint32_t asize,
                 neural::padding::type apadding,
                 float ak,
                 float aalpha,
                 float abeta)
    : engine (aengine)
    , output({ aoutput })
    , output_offset (aoutput_offset)
    , output_size (aoutput_size)
    , input({ ainput })
    , input_offset (ainput_offset)
    , size (asize)
    , padding (apadding)
    , k (ak)
    , alpha (aalpha)
    , beta (abeta) {}


// creates primitive with convolution implementation that supports provided arguments
primitive normalization::response::create(response::arguments arg) {
    return is_a_primitive::create<response>(arg);
}

}
