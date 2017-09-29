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
#include "assign_patch_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"

namespace cldnn
{
primitive_type_id assign_patch_type_id()
{
    static primitive_type_base<assign_patch> instance;
    return &instance;
}

layout assign_patch_inst::calc_output_layout(assign_patch_node const& node)
{
    auto source_layout = node.input().get_output_layout();
    auto nn_layout = node.nn().get_output_layout();

    auto result_sizes = tensor(1, source_layout.size.feature[0], nn_layout.size.spatial[0] + source_layout.size.spatial[0] - 1,
        nn_layout.size.spatial[1] + source_layout.size.spatial[1] - 1);
    auto result = layout({ source_layout.data_type, source_layout.format, result_sizes });

    return result;
}

std::string assign_patch_inst::to_string(assign_patch_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto& source                = node.input();
    auto& nn                    = node.nn();
    auto activation = desc->with_activation ? " true" : "false";

    primitive_description << "id: " << desc->id << ", type: assign_patch" << 
        "\n\tsource: " << source.id() << ", count: " << source.get_output_layout().count() << ",  size: " << source.get_output_layout().size <<
        "\n\tnn: " << nn.id() << ", count: " << nn.get_output_layout().count() << ",  size: " << nn.get_output_layout().size <<
        "\n\twith activation: " << activation << ", slope: " << desc->activation_negative_slope <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

assign_patch_inst::typed_primitive_inst(network_impl& network, assign_patch_node const& node)
    :parent(network, node)
{
    auto input_layout = input_memory(0).get_layout();
    auto nn_layout = input_memory(1).get_layout();

    CLDNN_ERROR_NOT_EQUAL(node.id(), "nn batch[0]", nn_layout.size.batch[0], "expected size of batch", 1, "Incorrect nn batch size.");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "nn feature[0]", nn_layout.size.feature[0], "expected size of feature", 1, "Incorrect nn feature size.");

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "input format", input_layout.format.value, "nn format", nn_layout.format);
}
}
