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
#include "reorder_arg.h"
#include "primitive_type_base.h"
#include <memory>

namespace cldnn
{
reorder_arg::reorder_arg(network_impl& network, std::shared_ptr<const reorder> desc)
    : primitive_arg_base(network, desc, desc->output_layout)
{
    auto& input_mem = input_memory(0);

    if (input_mem.argument().size.raw.size() != _output.argument().size.raw.size())
        //            throw std::runtime_error("Number of dimensions in reorder does not match. Meybe you want to use reshape primitive?"); //todo reshape
        throw std::runtime_error("Number of dimensions in reorder does not match.");
    if (!desc->substract_per_feature.empty())
    {
        if (input_mem.argument().size.feature.size() > 1)
        {
            throw std::runtime_error("Subtracting values work only for formats that have feature dimension == 1");
        }
        if (static_cast<size_t>(input_mem.argument().size.feature[0]) != desc->substract_per_feature.size())
            throw std::runtime_error("Number of features/channels in input does not match the number of features/channels in values to subtract");
    }
    auto input_padding_size = desc->input_padding().size();
    if(std::any_of(
        std::begin(input_padding_size.raw),
        std::end(input_padding_size.raw),
        [](int32_t p) { return p != 0; }))
    {
        throw std::runtime_error("Reorder with input which contains padding is NOT IMPLEMENTED yet!");
    }
}

primitive_type_id reorder_type_id()
{
    static primitive_type_base<reorder, reorder_arg> instance;
    return &instance;
}

}
