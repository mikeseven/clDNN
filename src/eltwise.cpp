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
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id eltwise_type_id()
{
    static primitive_type_base<eltwise> instance;
    return &instance;
}

layout eltwise_inst::calc_output_layout(eltwise_node const& node)
{
    return node.input().get_non_padded_output_layout();
}

std::string eltwise_inst::to_string(eltwise_node const& node)
{
    auto node_info  = node.desc_to_json();
    auto desc       = node.get_primitive();
    auto activation = desc->with_activation ? " true" : "false";
    auto& input_1   = node.input();
    auto& input_2   = node.input2();

    std::stringstream primitive_description;
    std::string       str_mode;

    switch(desc->mode)
    {
    case eltwise_mode::sum:
            str_mode = "sum";
            break;
    case eltwise_mode::sub:
            str_mode = "subtract";
            break;
    case eltwise_mode::max:
            str_mode = "max";
            break;
    case eltwise_mode::prod:
            str_mode = "product";
            break;
    default:
            str_mode = "not supported mode";
            break;
    }

    json_composite eltwise_info;
    eltwise_info.add("input_1", input_1.id());
    eltwise_info.add("input_2", input_2.id());
    eltwise_info.add("mode", str_mode);
    if (desc->with_activation)
    {
        eltwise_info.add("with activation", activation);
        eltwise_info.add("slope", desc->activation_negative_slope);
    }
    node_info.add("eltwise info", eltwise_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

eltwise_inst::typed_primitive_inst(network_impl& network, eltwise_node const& node)
    :parent(network, node)
{
    auto input_layout = input_memory(0).get_layout();
    auto input2_layout = input_memory(1).get_layout();

    // layout should be the same except padding, which can be different
    input_layout.data_padding = input2_layout.data_padding = {};

    CLDNN_ERROR_LAYOUT_MISMATCH(node.id(), "input layout", input_layout, "input_2 layout", input2_layout, "Different layouts of eltwise's inputs");
}
}
