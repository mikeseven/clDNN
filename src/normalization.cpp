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

#include "normalization_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{
primitive_type_id normalization_type_id()
{
    static primitive_type_base<normalization, normalization_inst> instance;
    return &instance;
}

layout normalization_inst::calc_output_layout(normalization_node const& node)
{
    return node.input().get_output_layout();
}

std::string normalization_inst::to_string(normalization_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto input                  = node.input();
    auto norm_size              = desc->size;
    auto k                      = desc->k;
    auto alpha                  = desc->alpha;
    auto beta                   = desc->beta;
    auto norm_region            = desc->norm_region == cldnn_lrn_norm_region::cldnn_lrn_norm_region_across_channel ? "across channel" : "within channel";

    primitive_description << "id: " << desc->id << ", type: normalization" << 
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
        "\n\tk: "     << k << ", alpha: " << alpha << ", beta: " << beta <<
        "\n\tsize of normalization: " << norm_size << ", normalization region: " << norm_region <<
        "\n\tinput padding: " << desc->input_padding <<
        "\n\toutput padding: " << desc->output_padding <<
        "\n\toutput: size: " << node.get_output_layout().size << '\n';
   
    return primitive_description.str();
}

normalization_inst::typed_primitive_inst(network_impl& network, normalization_node const& desc)
    :parent(network, desc)
{
    if (argument.size == 0)
    {
        throw std::runtime_error("LRN size must be greater than 0!");
    }
}
}
