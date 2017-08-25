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

#include "scale_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id scale_type_id()
{
    static primitive_type_base<scale> instance;
    return &instance;
}

layout scale_inst::calc_output_layout(scale_node const& node)
{
    auto result = node.input().get_output_layout();

    auto scale_sizes = node.scale_in().get_output_layout().size;
    auto input_sizes = result.size;

    auto scale_x_size = scale_sizes.spatial[0];
    auto scale_y_size = scale_sizes.spatial[1];

    auto input_x_size = input_sizes.spatial[0];
    auto input_y_size = input_sizes.spatial[1];

    if (scale_x_size != 1)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Scale x size", scale_x_size, "input x size", input_x_size, "");
    }
    if (scale_y_size != 1)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Scale y size", scale_y_size, "input y size", input_y_size, "");
    }
            
    return result;
}

std::string scale_inst::to_string(scale_node const& node)
{
    std::stringstream    primitive_description;
    auto node_info       = node.desc_to_json();
    auto desc            = node.get_primitive();
    auto& input          = node.input();
    auto& scale_input    = node.scale_in();

    json_composite scale_info;
    scale_info.add("input", input.id());
    scale_info.add("scale input", scale_input.id());

    node_info.add("scale info", scale_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

scale_inst::typed_primitive_inst(network_impl& network, scale_node const& node)
    :parent(network, node)
{
    auto scale_format = scale_memory().get_layout().format;

    auto scale_batch_size = scale_memory().get_layout().size.batch[0];
    auto scale_feature_size = scale_memory().get_layout().size.feature[0];

    auto input_batch_size = input_memory().get_layout().size.batch[0];
    auto input_feature_size = input_memory().get_layout().size.feature[0];

    if(scale_batch_size != 1)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Scale batch size", scale_batch_size, "input batch size", input_batch_size, "");
    }

    if (scale_feature_size != 1)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Scale feature size", scale_feature_size, "input feature size", input_feature_size, "");
    }

    if (!argument.bias.empty())
    {
        auto bias_format = bias_memory().get_layout().format;
        auto bias_raw_sizes = bias_memory().get_layout().size.raw;

        CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Scale format", scale_format.value, "bias format", bias_format);

        for (size_t i = 0; i < bias_memory().get_layout().size.raw.size(); ++i)
        {
            if (scale_memory().get_layout().size.raw[i] != bias_raw_sizes[i])
                CLDNN_ERROR_MESSAGE(node.id(), "Scale input size do not match bias size! Size index:" + std::to_string(i));
        }
    }
}
}
