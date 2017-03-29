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
#include "network_impl.h"
#include "primitive_type_base.h"

#include <algorithm>

namespace cldnn
{
primitive_type_id depth_concatenate_type_id()
{
    static primitive_type_base<depth_concatenate, depth_concatenate_inst> instance;
    return &instance;
}

layout depth_concatenate_inst::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const depth_concatenate> desc)
{
    auto& input_ids = desc->input;
    auto input0_desc = topology_map.at(input_ids.at(0))->primitive_desc;
    auto input_layout = input0_desc->type->calc_output_layout(topology_map, input0_desc);
    auto result_sizes = input_layout.size.sizes();
    auto input_format = input_layout.size.format;

    // get indicies of feature coordinates and initialize particular result coordinate to 0
    auto& format_order = input_format.order();
    assert(result_sizes.size() == format_order.size());
    if (input_layout.size.feature.size() != 1) throw std::domain_error("depth_concatenate supports only one feature dimension");

    auto feature_index = format_order.find_first_of(format_traits::feature_chars());
    assert(feature_index != std::string::npos);

    // calculate sum of features from all inputs
    result_sizes[feature_index] = 0;
    for(auto id : input_ids)
    {
        auto input_desc = topology_map.at(id)->primitive_desc;
        auto input_sizes = input_desc->type->calc_output_layout(topology_map, input_desc).size.sizes();
        result_sizes[feature_index] += input_sizes[feature_index];
    }
    return layout{input_layout.data_type, {input_format, result_sizes}};
}

depth_concatenate_inst::typed_primitive_inst(network_impl& network, std::shared_ptr<const depth_concatenate> desc)
    :parent(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{
    auto input_format = input_memory(0).get_layout().size.format;
    auto output_format = output_memory().get_layout().size.format;

    tensor::value_type depth_count = 0;
    auto input_size = _inputs.at(0)->non_padded_output_layout().size;
    auto output_size = non_padded_output_layout().size;
    for (const auto& i : _inputs)
    {
        auto& input_mem = i->output_memory();
        auto input_mem_size = i->non_padded_output_layout().size;
        if (input_mem.get_layout().size.format != input_format) throw std::runtime_error("Every input must have the same format!");
        if (input_mem_size.batch[0] != input_size.batch[0]) throw std::runtime_error("Every input must have the same number of batches!");
        if (input_mem_size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Every input must have the same size in X dimension!");
        if (input_size.spatial.size() > 1)
            if (input_mem_size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Every input must have the same size in Y dimension!");
        depth_count += input_mem.get_layout().size.feature[0];
    }

    if (output_format != input_format) throw std::runtime_error("Input and output must have the same format!");
    if (depth_count != output_size.feature[0]) throw std::runtime_error("Output depth count mismatch sum of input depths!");
    if (output_size.batch[0] != input_size.batch[0]) throw std::runtime_error("Output batch size must match input batch size!");
    if (output_size.spatial[0] != input_size.spatial[0]) throw std::runtime_error("Output X size must match input X size!");
    if (input_size.spatial.size() > 1)
        if (output_size.spatial[1] != input_size.spatial[1]) throw std::runtime_error("Output Y size must match input Y size!");
}
}
