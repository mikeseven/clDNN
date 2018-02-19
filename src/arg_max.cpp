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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "arg_max_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
	primitive_type_id arg_max_type_id()
	{
		static primitive_type_base<arg_max> instance;
		return &instance;
	}

	layout arg_max_inst::calc_output_layout(arg_max_node const& node)
	{
		auto desc = node.get_primitive();

		auto input_layout = node.input().get_output_layout();

		if (desc->output_max_value)
			return layout{ input_layout.data_type, input_layout.format, tensor{input_layout.size.batch[0], 1, (int32_t)(2 * desc->top_k), 0} };
		else
			return layout{ input_layout.data_type, input_layout.format, tensor{ input_layout.size.batch[0], 1, (int32_t)desc->top_k, 0 } };
	}

	std::string arg_max_inst::to_string(arg_max_node const& node)
	{
		auto desc = node.get_primitive();
		auto node_info = node.desc_to_json();
		auto activation = desc->with_activation ? "true" : "false";
		auto axis = desc->with_axis ? "true" : "false";
		auto max_val = desc->output_max_value ? "true" : "false";

		std::stringstream primitive_description;

		json_composite conv_info;
		conv_info.add("top_k", desc->top_k);
		conv_info.add("with axis", axis);
		if (desc->with_axis)
			conv_info.add("axis", desc->axis);
		conv_info.add("output max value", max_val);
		conv_info.add("with activation", activation);
		conv_info.add("slope", desc->activation_negative_slope);
		node_info.add("arg_max info", conv_info);
		node_info.dump(primitive_description);

		return primitive_description.str();
	}

	arg_max_inst::typed_primitive_inst(network_impl& network, arg_max_node const& node)
		: parent(network, node)
	{
		auto output_size = output_memory().get_layout().size;

		auto input_inst = input_memory().get_layout();
		auto output_inst = output_memory().get_layout();
		//TODO add some tests
	}
}
