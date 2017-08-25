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

#include "split_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"
#include "error_handler.h"

namespace cldnn
{
primitive_type_id split_type_id()
{
    static primitive_type_base<split> instance;
    return &instance;
}

layout split_inst::calc_output_layout(split_node const& node)
{
    auto output_ids = node.get_primitive()->output_ids;
    auto output_offsets = node.get_primitive()->output_offsets;
    auto param_num = output_ids.size();
    auto input_sizes = node.get_dependency(0).get_output_layout().size;

    //check if output_ids count equals output_offsets count
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Output_ids count", param_num, "output_offsets count", output_offsets.size(), "Output_ids count/ output_offsets count mismatch");

    for (decltype(param_num) i = 0; i < param_num; i++)
    {
        if (i != param_num - 1)
            //check if output offset sizes is less than next output offset sizes
            CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(), "output_offsets", output_offsets[i], "next output_offsets", output_offsets[i + 1], "Output_offsets tensor/ next input output_offsets tensor mismatch");
        else
            //check if output offset sizes matches output offsets sizes
            CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(), "Output_offsets", output_offsets[i], "input sizes", input_sizes, "Output_offsets tensor/ input tensor mismatch");

        //check if offsets do not extend input sizes and if match the output sizes
        CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(), "Output_offsets", output_offsets[i], "0 value", { 0, 0, 0, 0 }, "Invalid output_offsets: dims cannot be less than 0");
    }

    return node.input().get_output_layout();
}

std::string split_inst::to_string(split_node const& node)
{
    std::stringstream               primitive_description;
    auto desc                       = node.get_primitive();
    auto& input                     = node.input();
    auto output_ids                 = desc->output_ids;
    auto output_offsets             = desc->output_offsets;
    
    primitive_description << "id: " << desc->id << ", type: split" << 
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ",  size: " << input.get_output_layout().size <<
        "\n\toutput_ids count: " << output_ids.size() <<
        "\n\toffsets count: " << output_offsets.size() <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

split_inst::typed_primitive_inst(network_impl& network, split_node const& node)
    :parent(network, node)
{
}

}