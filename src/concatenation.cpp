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

#include "concatenation_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"

namespace cldnn
{
primitive_type_id concatenation_type_id()
{
    static primitive_type_base<concatenation> instance;
    return &instance;
}

layout concatenation_inst::calc_output_layout(concatenation_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;
    auto result_sizes = input_layout.size.sizes();

    auto axis_index = node.get_primitive()->axis;

    // calculate sum of features from all inputs
    result_sizes[axis_index] = 0;
    for (size_t i = 0; i < desc->input.size(); ++i)
    {
        auto input_sizes = node.input(i).get_output_layout().size.sizes();
        result_sizes[axis_index] += input_sizes[axis_index];
    }

    return layout{ input_layout.data_type, input_format, result_sizes };
}

std::string concatenation_inst::to_string(concatenation_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    std::stringstream           ss_inputs;
    for (size_t i = 0; i < node.inputs_count(); ++i)
    {
        ss_inputs << node.input(i).id();
        ss_inputs << ", count: " << node.input(i).get_output_layout().count();
        i != (node.inputs_count() - 1) ? ss_inputs << ", " : ss_inputs << "";
    }

    primitive_description << "id: " << desc->id << ", type: depth_concatenate" << 
        "\n\tconcat axis: " << desc->axis <<
        "\n\tinputs count: " << node.inputs_count() << 
        "\n\tinputs: " << ss_inputs.str() << 
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

concatenation_inst::typed_primitive_inst(network_impl& network, concatenation_node const& node)
    :parent(network, node)
{
    auto input_format = input_memory(0).get_layout().fused_format();
    auto output_format = output_memory().get_layout().fused_format();

    tensor::value_type concat_count = 0;
    auto input_size = input_memory(0).get_layout().size;;
    auto output_size = output_memory().get_layout().size;
    for (const auto& i : _deps)
    {
        auto& input_mem = i->output_memory();
        auto input_mem_size = input_mem.get_layout().size;
        for (int dim = concatenation::along_b; dim <= concatenation::along_y; ++dim)
        {
            if (dim == node.get_primitive()->axis)
                concat_count += input_mem_size.raw[dim];
            else
            {
                CLDNN_ERROR_NOT_EQUAL(node.id(), "Input size dim: " + std::to_string(dim), input_size.raw[dim], "input memory dim: " + std::to_string(dim), input_mem_size.raw[dim], "Every input must have the same size");
            }
        }
    }

    CLDNN_ERROR_NOT_EQUAL(node.id(), "Output format (fused) ", output_format, "input format (fused)", input_format, "Fused input/output formats mistmach");

    for (int dim = concatenation::along_b; dim <= concatenation::along_y; ++dim)
    {
        if (dim == node.get_primitive()->axis)
        {
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Concat count", concat_count, "output size dim:" + std::to_string(dim), output_size.raw[dim], "Output size in concatenated dimension mismatch sum of inputs!");
        }
        else
        {
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Input size dim: " + std::to_string(dim), input_size.raw[dim], "output size dim:" + std::to_string(dim), output_size.raw[dim], "Output size in non-concatenated dimension mistmatch input");
        }
    }

    if (node.can_be_optimized())
        for (auto const& i : _deps)
            i->_output = _output;
}
}
