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
#include "prior_box.h"
#include "../primitive.hpp"

namespace cldnn
{

struct prior_box : public primitive_base<prior_box, CLDNN_PRIMITIVE_DESC(prior_box)>
{
    CLDNN_DECLATE_PRIMITIVE(prior_box)

    prior_box(
        const primitive_id& id,
        const primitive_id& input,
		const tensor& img_size,
		const std::vector<float>& min_sizes,
		const std::vector<float>& max_sizes = {},
		const std::vector<float>& aspect_ratios = {},
		const bool flip = true,
		const bool clip = false,
		const std::vector<float>& variance = {},
		const float step_width = 0.f,
		const float step_height = 0.f,
		const float offset = 0.5f,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input}, input_padding, output_padding, img_size, cldnn_float_arr{ nullptr, 0 }, cldnn_float_arr{ nullptr, 0 }, cldnn_float_arr{ nullptr, 0 },
			flip, clip, cldnn_float_arr{ nullptr, 0 }, step_width, step_height, offset)
		, img_size(img_size)
        , min_sizes(min_sizes)
        , max_sizes(max_sizes)
		, flip(flip)
		, clip(clip)
		, step_width(step_width)
		, step_height(step_height)
		, offset(offset)
    {
		this->aspect_ratios.push_back(1.f);
		for (auto new_aspect_ratio : aspect_ratios) {
			bool already_exist = false;
			for (auto aspect_ratio : this->aspect_ratios) {
				if (fabs(new_aspect_ratio - aspect_ratio) < 1e-6) {
					already_exist = true;
					break;
				}
			}
			if (!already_exist) {
				this->aspect_ratios.push_back(new_aspect_ratio);
				if (flip) {
					this->aspect_ratios.push_back(1.f / new_aspect_ratio);
				}
			}
		}
		if (variance.size() > 1) {
			for (size_t i = 0; i < variance.size(); ++i) {
				this->variance.push_back(variance[i]);
			}
		}
		else if (variance.size() == 1) {
			this->variance.push_back(variance[0]);
		}
		else {
			// Set default to 0.1.
			this->variance.push_back(0.1f);
		}

		_dto.min_sizes = float_vector_to_arr(this->min_sizes);
		_dto.max_sizes = float_vector_to_arr(this->max_sizes);
		_dto.aspect_ratios = float_vector_to_arr(this->aspect_ratios);
		_dto.variance = float_vector_to_arr(this->variance);
	}

	prior_box(const dto* dto)
        : primitive_base(dto)
		, img_size(_dto.img_size)
        , min_sizes(float_arr_to_vector(_dto.min_sizes))
        , max_sizes(float_arr_to_vector(_dto.max_sizes))
		, aspect_ratios(float_arr_to_vector(_dto.aspect_ratios))
		, flip(_dto.flip != 0)
		, clip(_dto.clip != 0)
		, variance(float_arr_to_vector(_dto.variance))
		, step_width(_dto.step_width)
		, step_height(_dto.step_height)
		, offset(_dto.offset)
    {}

	const tensor img_size;
	const std::vector<float> min_sizes;
	const std::vector<float> max_sizes;
	std::vector<float> aspect_ratios;
	const bool flip;
	const bool clip;
	std::vector<float> variance;
	const float step_width;
	const float step_height;
	const float offset;

private:

	static cldnn_float_arr float_vector_to_arr(const std::vector<float>& stor)
	{
		return{ stor.data(), stor.size() };
	}

	static std::vector<float> float_arr_to_vector(const cldnn_float_arr& arr)
	{
		std::vector<float> result(arr.size);
		for (size_t i = 0; i < arr.size; i++)
		{
			result[i] = arr.data[i];
		}
		return result;
	}
};

}