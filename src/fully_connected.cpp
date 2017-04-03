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
#include "fully_connected_inst.h"
#include "primitive_type_base.h"
namespace cldnn
{
primitive_type_id fully_connected_type_id()
{
    static primitive_type_base<fully_connected, fully_connected_inst> instance;
    return &instance;
}

namespace
{
bool is_batch_after_spatial(const std::string order)
{
    bool spatial_found = false;
    bool batch_found = false;
    for (auto c : order)
    {
        switch (c)
        {
        case 'b':
        case 'n':
            batch_found = true;
            if (spatial_found)
                return true;
        case 'x':
        case 'y':
        case 'z':
        case 'w':
        case 's':
            spatial_found = true;
            if (batch_found)
                return false;
        default: break;
        }
    }
    return false;
}
}

layout fully_connected_inst::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const fully_connected> desc)
{
    auto input_desc = topology_map.at(desc->input[0])->primitive_desc;
    auto input_layout = input_desc->type->calc_output_layout(topology_map, input_desc);
    
    auto weights_desc = topology_map.at(desc->weights)->primitive_desc;
    auto weights_layout = weights_desc->type->calc_output_layout(topology_map, weights_desc);

    if(is_batch_after_spatial(input_layout.size.format.order()) || 
        (input_layout.size.format == format::bfyx &&                //this condition tests whether our input is batch>1 in bfyx format, if yes there will be
        input_layout.size.batch[0] > 1))                            //extra reorder between input and this fc from bfyx to yxfb format (so "is_batch_after_spatial" should return true)
    {
        auto result = layout(input_layout.data_type, tensor(format::xb, { weights_layout.size.batch[0], input_layout.size.batch[0] }));
        return result;
    }
    else
    {
        auto result = layout(input_layout.data_type, tensor(format::bx, { input_layout.size.batch[0], weights_layout.size.batch[0] }));
        return result;
    }
}

fully_connected_inst::typed_primitive_inst(network_impl& network, std::shared_ptr<const fully_connected> desc)
    :parent(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{
    auto input_size = input_memory().get_layout().size;
    auto output_size = output_memory().get_layout().size;

    if(input_size.format != format::yxfb
        && input_size.format != format::bfyx //special batch1 case
        && (input_size.raw.size() != output_size.raw.size()) )
    {
        throw std::invalid_argument("Fully connected input/output number of dimension does not match.");
    }
}

}
