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

#include "normalize_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{
primitive_type_id normalize_type_id()
{
    static primitive_type_base<normalize, normalize_inst> instance;
    return &instance;
}

layout normalize_inst::calc_output_layout(normalize_node const& node)
{
    return node.input().get_output_layout();
}

std::string normalize_inst::to_string(normalize_node const& node)
{
	std::stringstream           primitive_description;
	auto desc = node.get_primitive();
	auto input = node.input();
	auto epsilon = desc->epsilon;
	auto scale = desc->scale_factor;
	auto norm_region = desc->across_spatial ? "across spatial" : "within spatial";

	primitive_description << "id: " << desc->id << ", type: normalize" <<
		"\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
		"\n\tepsilon: " << epsilon << ", scale factor: " << scale << ", normalization region: " << norm_region <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
		"\n\toutput: size: " << node.get_output_layout().size << '\n';

	return primitive_description.str();
}

normalize_inst::typed_primitive_inst(network_impl& network, normalize_node const& node)
    :parent(network, node)
{
	auto input_layout = node.input().get_output_layout();
	if (input_layout.format != format::bfyx)
	{
		throw std::invalid_argument("Normalize layer supports only bfyx input format.");
	}
}
}
