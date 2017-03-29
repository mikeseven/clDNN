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

#include "activation_inst.h"
#include "primitive_type_base.h"
#include <memory>

namespace cldnn
{
primitive_type_id activation_type_id()
{
    static primitive_type_base<activation, activation_inst> instance;
    return &instance;
}

layout activation_inst::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const activation> desc)
{
    auto input_desc = topology_map.at(desc->input[0])->primitive_desc;
    auto result = input_desc->type->calc_output_layout(topology_map, input_desc);
    return result;
}

activation_inst::typed_primitive_inst(network_impl& network, std::shared_ptr<const activation> desc)
    :parent(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{
    auto input_arg  = input_memory().get_layout();
    auto output_arg = output_memory().get_layout();
    
    if (input_arg.size.raw.size() != output_arg.size.raw.size())    throw std::runtime_error("ReLU input/output number of dimension does not match.");
}

}
