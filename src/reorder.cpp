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
#include "reorder_inst.h"
#include "primitive_type_base.h"
#include <memory>

namespace cldnn
{

primitive_type_id reorder_type_id()
{
    static primitive_type_base<reorder, reorder_inst> instance;
    return &instance;
}

reorder_inst::typed_primitive_inst(network_impl& network, std::shared_ptr<const reorder> desc)
    : parent(network, desc, desc->output_layout)
{
    auto& input_mem = input_memory();

    if (input_mem.get_layout().size.raw.size() < _output.get_layout().size.raw.size())
        throw std::runtime_error("Input dimension < output dimension. Reorder primitive woks only with same dimension sizes (reorder) or when input > output (flatten).");
    if (!desc->substract_per_feature.empty())
    {
        if (input_mem.get_layout().size.feature.size() > 1)
        {
            throw std::runtime_error("Subtracting values work only for formats that have feature dimension == 1");
        }
        if (static_cast<size_t>(input_mem.get_layout().size.feature[0]) != desc->substract_per_feature.size())
            throw std::runtime_error("Number of features/channels in input does not match the number of features/channels in values to subtract");
    }
    if (argument.input_padding)
    {
        throw std::runtime_error("Reorder with input which contains padding is NOT IMPLEMENTED yet!");
    }
}

}
