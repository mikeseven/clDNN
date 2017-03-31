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
#include "eltwise_inst.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
primitive_type_id eltwise_type_id()
{
    static primitive_type_base<eltwise, eltwise_inst> instance;
    return &instance;
}

layout eltwise_inst::calc_output_layout(eltwise_node const& node)
{
    return node.input().get_output_layout();
}

eltwise_inst::typed_primitive_inst(network_impl& network, eltwise_node const& node)
    :parent(network, node)
{
    auto input_layout = input_memory().get_layout();
    auto input2_layout = input2_memory().get_layout();

    if (input_layout != input2_layout)
    {
        throw std::runtime_error("Different layouts of eltwise's inputs");
    }
}
}
