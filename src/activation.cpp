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

#include "activation_arg.h"
#include "primitive_type_base.h"
#include <memory>

namespace cldnn
{
primitive_type_id activation::type_id()
{
    static primitive_type_base<activation, activation_arg> instance;
    return &instance;
}

layout activation_arg::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const activation> desc)
{
    auto input_desc = topology_map.at(desc->input()[0])->primitive_desc;
    auto result = input_desc->type()->calc_output_layout(topology_map, input_desc);
    return result;
}

activation_arg::activation_arg(network_impl& network, std::shared_ptr<const activation> desc)
    :primitive_arg_base(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{
    auto input_offset = desc->input_offset().transform(input_memory(0).get_layout().size.format, 0);
    auto output_offset = desc->output_offset().transform(output_memory().get_layout().size.format, 0);
    auto& output_size = output_memory().get_layout().size;
    
    auto input_arg  = input_memory(0).argument();
    auto output_arg = output_memory().argument();
    
    if (input_arg.size.raw.size() != output_arg.size.raw.size())    throw std::runtime_error("ReLU input/output number of dimension does not match.");
    for (auto x : input_offset.raw)  if (x < 0)                     throw std::runtime_error("ReLU negative input offset.");
    
    for (size_t i = 0; i < input_arg.size.raw.size(); ++i) {
        if (input_arg.size.raw[i]  < output_size.raw[i] + input_offset.raw[i]) throw std::runtime_error("ReLU input/output size does not match.");
        if (output_arg.size.raw[i] < output_size.raw[i] + output_offset.raw[i]) throw std::runtime_error("ReLU sizes to small.");
    }
}

}
