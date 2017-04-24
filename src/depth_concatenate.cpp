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

#include "depth_concatenate_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{
primitive_type_id depth_concatenate_type_id()
{
    static primitive_type_base<depth_concatenate, depth_concatenate_inst> instance;
    return &instance;
}

layout depth_concatenate_inst::calc_output_layout(depth_concatenate_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto result_sizes = input_layout.size.sizes();
    auto input_format = input_layout.size.format;

    // get indicies of feature coordinates and initialize particular result coordinate to 0
    auto& format_order = input_format.order();
    assert(result_sizes.size() == format_order.size());
    if (input_layout.size.feature.size() != 1)
        throw std::domain_error("depth_concatenate supports only one feature dimension");

    auto feature_index = format_order.find_first_of(format_traits::feature_chars());
    assert(feature_index != std::string::npos);

    // calculate sum of features from all inputs
    result_sizes[feature_index] = 0;
    for (size_t i = 0; i < desc->input.size(); ++i)
    {
        auto input_sizes = node.input(i).get_output_layout().size.sizes();
        result_sizes[feature_index] += input_sizes[feature_index];
    }

    return layout{ input_layout.data_type,{ input_format, result_sizes } };
}

std::string depth_concatenate_inst::to_string(depth_concatenate_node const& node)
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
        "\n\tinputs count: " << node.inputs_count() << 
        "\n\tinputs: " << ss_inputs.str() << 
        "\n\toutput padding: " << desc->output_padding <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

depth_concatenate_inst::typed_primitive_inst(network_impl& network, depth_concatenate_node const& node)
    :parent(network, node)
{
    auto input_format = input_memory(0).get_layout().fused_format();
    auto output_format = output_memory().get_layout().fused_format();

    tensor::value_type depth_count = 0;
    auto input_size = _deps.at(0)->non_padded_output_layout().size;
    auto output_size = non_padded_output_layout().size;
    for (const auto& i : _deps)
    {
        auto& input_mem = i->output_memory();
        auto input_mem_size = i->non_padded_output_layout().size;
        if (input_mem.get_layout().fused_format() != input_format)
            throw std::runtime_error("Every input must have the same format!");

        if (input_mem_size.batch[0] != input_size.batch[0])
            throw std::runtime_error("Every input must have the same number of batches!");

        if (input_mem_size.spatial[0] != input_size.spatial[0])
            throw std::runtime_error("Every input must have the same size in X dimension!");

        if (input_size.spatial.size() > 1)
            if (input_mem_size.spatial[1] != input_size.spatial[1])
                throw std::runtime_error("Every input must have the same size in Y dimension!");

        depth_count += input_mem.get_layout().size.feature[0];
    }

    if (output_format != input_format)
        throw std::runtime_error("Input and output must have the same format!");

    if (depth_count != output_size.feature[0])
        throw std::runtime_error("Output depth count mismatch sum of input depths!");

    if (output_size.batch[0] != input_size.batch[0])
        throw std::runtime_error("Output batch size must match input batch size!");

    if (output_size.spatial[0] != input_size.spatial[0])
        throw std::runtime_error("Output X size must match input X size!");

    if (input_size.spatial.size() > 1)
        if (output_size.spatial[1] != input_size.spatial[1])
            throw std::runtime_error("Output Y size must match input Y size!");
}
}
