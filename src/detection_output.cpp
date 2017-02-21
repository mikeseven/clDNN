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

#include "detection_output_inst.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
primitive_type_id detection_output_type_id()
{
    static primitive_type_base<detection_output, detection_output_inst> instance;
    return &instance;
}

layout detection_output_inst::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const detection_output> desc)
{
	auto inputs = desc->input;
	if (inputs.size() != 3) 
	{
		throw std::invalid_argument("Detection output layer must get 3 inputs.");
	}	
    auto input_desc = topology_map.at(desc->input[0])->primitive_desc;
    auto input_layout = input_desc->type->calc_output_layout(topology_map, input_desc);

	if (input_layout.data_type != data_types::f32)
	{
		throw std::invalid_argument("Detection output layer supports only FP32.");
	}

	// Batch size and feature size are 1.
	// Number of bounding boxes to be kept is set to keep_top_k*batch size. 
	// If number of detections is lower than keep_top_k, will write dummy results at the end with image_id=-1. 
	// Each row is a 7 dimension vector, which stores:
	// [image_id, label, confidence, xmin, ymin, xmax, ymax]
    return{ input_layout.data_type, cldnn::tensor(cldnn::format::bfyx,{ 1, 1, desc->keep_top_k * input_layout.size.batch[0], DETECTION_OUTPUT_ROW_SIZE }) };
}

detection_output_inst::typed_primitive_inst(network_impl& network, std::shared_ptr<const detection_output> desc)
    :parent(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
{
	if ( (location_memory().get_layout().size.format != format::bfyx) ||
		 (confidence_memory().get_layout().size.format != format::bfyx) ||
		 (prior_box_memory().get_layout().size.format != format::bfyx) )
	{
		throw std::invalid_argument("Detection output layer supports only bfyx input format.");
	}
	if ((location_memory().get_layout().data_type != data_types::f32) ||
		(confidence_memory().get_layout().data_type != data_types::f32) ||
		(prior_box_memory().get_layout().data_type != data_types::f32))
	{
		throw std::invalid_argument("Detection output layer supports only FP32.");
	}

	if (argument.input_padding || argument.output_padding)
	{
		throw std::invalid_argument("Detection output layer doesn't support input and output padding.");
	}
}
}
