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
#include "normalize.h"
#include "../primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Normalizes the input using an L2 norm and multiplies the output with scale_factor.
/// @details The L2 norm is computed as:<br>
/// Across spatial mode (across_spatial=true)-<br>
/// norm(i,x,y) = sqrt( &Sigma;( in(f,w,h)^2 ) + epsilon ) where f in range (0,num_of_features), w in range (0,input_width), h in range (0,input_height).<br>
/// The summation is performed over all the pixels in the batch.<br>
/// Within spatial mode (across_spatial=false)-<br>
/// norm(i,x,y) = sqrt( &Sigma;( in(f,x,y)^2 ) + epsilon ) where f in range (0,num_of_features).<br>
/// The summation is performed over this (x,y) position on all the features.<br>
/// @par Algorithm:
///   out(i,x,y) = ( in(i,x,y) / norm(i,x,y) ) * scale_factor 
/// @par Where:
///   @li out(i,x,y) : value at x, y from i-th feature map after normalization.
///   @li in(i,x,y) : value at x, y from i-th feature map before normalization.
///   @li norm(i,x,y) : L2 norm as described above.
///   @li scale_factor : parameter for the layer.
struct normalize :public primitive_base<normalize, CLDNN_PRIMITIVE_DESC(normalize)>
{
    CLDNN_DECLATE_PRIMITIVE(normalize)

    /// @brief Constructs normalize primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param across_spatial Determines if the normalization is done across or within spatial (see documentation above).
	/// @param epsilon Epsilon for not dividing by zero while normalizing.
	/// @param scale_factor Scales the output of the normalization.
	normalize(
        const primitive_id& id,
        const primitive_id& input,
		const bool across_spatial = true,
		const float epsilon = 1e-10f,
		const float scale_factor = 1.f,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input}, input_padding, output_padding)
        , across_spatial(across_spatial)
        , epsilon(epsilon)
		, scale_factor(scale_factor)
    {}

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{normalize}
	normalize(const dto* dto)
        : primitive_base(dto)
        , across_spatial(dto->across_spatial != 0)
        , epsilon(dto->epsilon)
        , scale_factor(dto->scale_factor)
    {}

    /// @brief Determines if the normalization is done across or within spatial (see documentation above).
	bool across_spatial;
	/// @brief Epsilon for not dividing by zero while normalizing.
	float epsilon;
	/// @brief Scales the output of the normalization.
	float scale_factor;

protected:
    void update_dto(dto& dto) const override
    {
        dto.across_spatial = across_spatial;
		dto.epsilon = epsilon;
		dto.scale_factor = scale_factor;
    }
};
/// @}
/// @}
/// @}
}