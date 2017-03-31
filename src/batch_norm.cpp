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

#include "batch_norm_inst.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
primitive_type_id batch_norm_type_id()
{
    static primitive_type_base<batch_norm, batch_norm_inst> instance;
    return &instance;
}

layout batch_norm_inst::calc_output_layout(batch_norm_node const& node)
{
    return node.input().get_output_layout();
}

batch_norm_inst::typed_primitive_inst(network_impl& network, batch_norm_node const& node)
    :parent(network, node) 
{
    auto input_format = input_memory().get_layout().size.format;
    auto output_format = output_memory().get_layout().size.format;
    auto mean_format = mean_memory().get_layout().size.format;
    auto variance_format = variance_memory().get_layout().size.format;

    if (input_format != format::yxfb)
    {
        throw std::runtime_error("Batch norm input is not in yxfb format!");
    }
    if (output_format != format::yxfb)
    {
        throw std::runtime_error("Batch norm output is not in yxfb format!");
    }
    if (mean_format != format::yxfb &&
        mean_format != format::bfyx)
    {
        throw std::runtime_error("Mean is not in yxfb or bfyx format!");
    }
    if (variance_format != format::yxfb &&
        variance_format != format::bfyx)
    {
        throw std::runtime_error("Variance is not in yxfb or bfyx format!");
    }
}
}