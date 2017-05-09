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

#include "crop_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{
primitive_type_id crop_type_id()
{
    static primitive_type_base<crop, crop_inst> instance;
    return &instance;
}

layout crop_inst::calc_output_layout(crop_node const& node)
{
    return node.reference_input().get_output_layout();
}

std::string crop_inst::to_string(crop_node const& node)
{
    std::stringstream               primitive_description;
    auto desc                       = node.get_primitive();
    auto input                      = node.input();
    auto ref_input                  = node.reference_input();
    auto offsets                    = desc->offsets;
    
    primitive_description << "id: " << desc->id << ", type: crop" << 
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ",  size: " << input.get_output_layout().size <<
        "\n\treference input: " << ref_input.id() << ", sizes: " << ref_input.get_output_layout().size <<
        "\n\toffsets: " << offsets <<
        "\n\tinput padding: " << desc->input_padding <<
        "\n\toutput padding: " << desc->output_padding <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

crop_inst::typed_primitive_inst(network_impl& network, crop_node const& node)
    :parent(network, node)
{
    auto reference_input_sizes = reference_input_memory().get_layout().size;
    auto reference_format = reference_input_memory().get_layout().format;
    auto input_sizes = input_memory().get_layout().size;
    auto input_format = input_memory().get_layout().format;
    auto offsets = argument.offsets;

    if (input_format != reference_format)
        throw std::runtime_error("Mismatch between input and reference_input format order!");

    if ((input_format!= format::yxfb) && (input_format != format::bfyx))
        throw std::runtime_error("Crop layer is only supported for yxfb and bfyx formats!");

    //check if output sizes matches reference input sizes
    if (reference_input_sizes.batch[0] > input_sizes.batch[0])
        throw std::runtime_error("Reference input batch dimension > input batch dimension!");
    if (reference_input_sizes.feature[0] > input_sizes.feature[0])
        throw std::runtime_error("Reference input feature dimension > input batch dimension!");
    if (reference_input_sizes.spatial[0] > input_sizes.spatial[0])
        throw std::runtime_error("Reference input X dimension > input batch dimension!");
    if (reference_input_sizes.spatial[1] > input_sizes.spatial[1])
        throw std::runtime_error("Reference input Y dimension > input batch dimension!");

    //check if offsets do not extend input sizes and if match the output sizes
    if (((offsets.batch[0] < 0) || (input_sizes.batch[0] - offsets.batch[0]) < reference_input_sizes.batch[0]))
        throw std::runtime_error("Invalid Batch offset: negative value or exceeds data for output!");
    if (((offsets.feature[0] < 0) || (input_sizes.feature[0] - offsets.feature[0]) < reference_input_sizes.feature[0]))
        throw std::runtime_error("Invalid Feature offset: negative value or exceeds data for output!");
    if (((offsets.spatial[0] < 0) || (input_sizes.spatial[0] - offsets.spatial[0]) < reference_input_sizes.spatial[0]))
        throw std::runtime_error("Invalid X offset: negative value or exceeds data for output!");
    if (((offsets.spatial[1] < 0) || (input_sizes.spatial[1] - offsets.spatial[1]) < reference_input_sizes.spatial[1]))
        throw std::runtime_error("Invalid Y offset: negative value or exceeds data for output!");
}
}