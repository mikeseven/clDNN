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
#include "network_impl.h"

namespace cldnn
{
primitive_type_id crop_type_id()
{
    static primitive_type_base<crop, crop_inst> instance;
    return &instance;
}

layout crop_inst::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const crop> desc)
{
    auto reference_input_desc = topology_map.at(desc->reference_input)->primitive_desc;
    auto result = reference_input_desc->type->calc_output_layout(topology_map, reference_input_desc);
    return result;
}

crop_inst::typed_primitive_inst(network_impl& network, std::shared_ptr<const crop> desc)
    :parent(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{
    auto reference_input_sizes = reference_input_memory().get_layout().size;
    auto reference_format = reference_input_sizes.format;
    auto input_sizes = input_memory().get_layout().size;
    auto input_format = input_sizes.format;
    auto offsets = desc->offsets;

    if (input_format != reference_format)
        throw std::runtime_error("Mismatch between input and reference_input format order!");

    if (offsets.format != input_format)
        throw std::runtime_error("Mismatch between offsets and input format order!");

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