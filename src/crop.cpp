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
#include "memory_impl.h"
#include "error_handler.h"

namespace cldnn
{
primitive_type_id crop_type_id()
{
    static primitive_type_base<crop> instance;
    return &instance;
}

layout crop_inst::calc_output_layout(crop_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    auto result = layout({ input_layout.data_type, input_layout.format, node.get_primitive()->reference_input });
    return result;
}

std::string crop_inst::to_string(crop_node const& node)
{
    std::stringstream               primitive_description;
    auto desc                       = node.get_primitive();
    auto& input                     = node.input();
    auto ref_input                  = desc->reference_input;
    auto offsets                    = desc->offsets;
    
    primitive_description << "id: " << desc->id << ", type: crop" << 
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ",  size: " << input.get_output_layout().size <<
        "\n\treference input sizes: " << ref_input <<
        "\n\toffsets: " << offsets <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

crop_inst::typed_primitive_inst(network_impl& network, crop_node const& node)
    :parent(network, node)
{
    auto reference_input_sizes = argument.reference_input;
    auto input_sizes = input_memory().get_layout().size;
    auto input_format = input_memory().get_layout().format;
    auto offsets = argument.offsets;

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Input format", input_format.value, "supported crop input formats", format::yxfb, format::bfyx );

    //check if output sizes matches reference input sizes
    CLDNN_ERROR_TENSOR_SIZES_GREATER_THEN(node.id(), "Reference input", reference_input_sizes, "input sizes", input_sizes, "Reference input tensor/ input tensor mismtach");
    
    //check if offsets do not extend input sizes and if match the output sizes
    CLDNN_ERROR_TENSOR_SIZES_LESS_THEN(node.id(), "Batch offsets", offsets, "0 value", { 0, 0, 0, 0 }, "Invalid Batch offset: negative value");
    auto input_size_sub_offsets = input_sizes - offsets;
    CLDNN_ERROR_TENSOR_SIZES_LESS_THEN(node.id(), "input sizes - offsets", input_size_sub_offsets, "reference input sizes", reference_input_sizes, "Invalid Batch offset: exceeds data for output!");

    if (node.can_be_optimized())
    {
        reuse_input();
    }
}


void crop_inst::on_execute()
{
    if (!node.can_be_optimized())
        return;

    if (_output && _output->is_the_same_buffer(input_memory()))
        return;

    reuse_input();
}

void crop_inst::reuse_input()
{
    _output = api_cast(_network.get_engine()->reinterpret_buffer(api_cast(input_memory().get()), node.get_output_layout()));
}
}