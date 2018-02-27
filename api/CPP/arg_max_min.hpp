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
#pragma once
#include "../C/arg_max_min.h"
#include "primitive.hpp"

namespace cldnn
{
	/// @addtogroup cpp_api C++ API
	/// @{
	/// @addtogroup cpp_topology Network Topology
	/// @{
	/// @addtogroup cpp_primitives Primitives
	/// @{

	/// @brief Finds the index of the k max values of input.
	/// Also supports built-in Relu @CLDNN_PRIMITIVE_DESC{activation} available by setting it in arguments.
	struct arg_max_min : public primitive_base<arg_max_min, CLDNN_PRIMITIVE_DESC(arg_max_min)>
	{
		CLDNN_DECLATE_PRIMITIVE(arg_max_min)

		/// @brief Enum type to specify output type - index of max or min values
		enum out_type
		{
			max,
			min,
		};

		/// @brief Constructs arg_max primitive.
		/// @param id This primitive id.
		/// @param input Input primitive id.
		/// @param top_k Number of maximum indexes to output.
		/// @param output_max_value Enables outputing vector of pairs (maximum index, maximum value).
		/// @param with_activation Enable Relu activation.
		/// @param activation_slp Relu activation slope.
		arg_max_min(
			const primitive_id& id,
			const primitive_id& input,
			out_type output_type,
			uint32_t top_k = 1,
			bool with_activation = false,
			float activation_slp = 0.0f,
			const padding& output_padding = padding()
		)
			:primitive_base(id, { input }, output_padding)
			, top_k(top_k)
			, output_type(output_type)
			, with_axis(false)
			, axis(0)
			, with_activation(with_activation)
			, activation_negative_slope(activation_slp)
		{}

		/// @brief Constructs arg_max primitive with axis to maximize along;
		/// @param id This primitive id.
		/// @param input Input primitive id.
		/// @param top_k Number of maximum indexes to output.
		/// @param output_max_value Enables outputing vector of pairs (maximum index, maximum value).
		/// @param with_activation Enable Relu activation.
		/// @param activation_slp Relu activation slope.
		arg_max_min(
			const primitive_id& id,
			const primitive_id& input,
			out_type output_type,
			uint32_t axis,
			uint32_t top_k = 1,
			bool with_activation = false,
			float activation_slp = 0.0f,
			const padding& output_padding = padding()
		)
			:primitive_base(id, { input }, output_padding)
			, top_k(top_k)
			, output_type(output_type)
			, with_axis(true)
			, axis(axis)
			, with_activation(with_activation)
			, activation_negative_slope(activation_slp)
		{}


		/// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{arg_max}
		arg_max_min(const dto* dto)
			:primitive_base(dto)
			, top_k(dto->top_k)
			, output_type(static_cast<out_type>(dto->output_type))
			, with_axis(dto->with_axis != 0)
			, axis(dto->axis)
			, with_activation(dto->with_activation != 0)
			, activation_negative_slope(dto->activation_negative_slope)
		{}

		/// @brief Enable Relu activation.
		bool with_activation;
		/// @brief Relu activation slope.
		float activation_negative_slope;
		/// @brief Number of maximal indexes to output.
		uint32_t top_k;
		/// @brief Type of output - max or mix.
		out_type output_type;
		/// @brief Indicates that the primitive has user defined axis to maximize along;
		bool with_axis;
		/// @brief Axis to maximize along. If not set, maximize the flattened trailing dimensions for each index of the batch dimension.
		uint32_t axis;

	protected:

		void update_dto(dto& dto) const override
		{
			dto.top_k = top_k;
			dto.output_type = static_cast<cldnn_arg_max_min_out>(output_type);
			dto.with_axis = with_axis;
			dto.axis = axis;
			dto.with_activation = with_activation;
			dto.activation_negative_slope = activation_negative_slope;
		}
	};
	/// @}
	/// @}
	/// @}
}