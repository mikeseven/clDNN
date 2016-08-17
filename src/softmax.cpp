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
namespace normalization {

softmax::arguments::arguments(neural::engine::type eng, primitive out, vector<uint32_t> out_off, vector<uint32_t> out_siz, primitive in, vector<int32_t> in_off)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    {}

softmax::arguments::arguments(neural::engine::type eng, primitive out, primitive in)
    : engine(eng)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
 {
        if (in.id() != type_id<const memory>()->id)
        {
            input = { in.output[0] };
            input_offset = (in.output[0].as<const memory&>().argument.size.batch.size(), in.output[0].as<const memory&>().argument.size.spatial.size(), in.output[0].as<const memory&>().argument.size.feature.size());
        }
        else
        {
            input = { in };
            input_offset = (in.as<const memory&>().argument.size.batch.size(), in.as<const memory&>().argument.size.spatial.size(), in.as<const memory&>().argument.size.feature.size());
        }
        //in[0].primitive.id() != type_id<const memory>()->id ? in[0].primitive.output[0] : in[0]
}

// creates primitive with softmax implementation that supports provided arguments
primitive softmax::create(softmax::arguments arg) {
    auto& input_offset  = arg.input_offset;
    auto& output_offset = arg.output_offset;
    auto& output_size   = arg.output_size;

    auto& input_arg  = arg.input[0].primitive().as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;
    for (auto &x : input_offset.raw) if (x < 0) throw std::runtime_error("Softmax negative input offset.");

    for(size_t i = 0; i < input_arg.size.raw.size(); ++i) {
        if( input_arg.size.raw[i] < output_size.raw[i] +  input_offset.raw[i]) throw std::runtime_error("Softmax input/output size does not match.");
        if(output_arg.size.raw[i] < output_size.raw[i] + output_offset.raw[i]) throw std::runtime_error("Softmax sizes too small.");
    }

    return is_a_primitive::create<softmax>(arg);
}

} // namespace normalization
} // namespace neural
