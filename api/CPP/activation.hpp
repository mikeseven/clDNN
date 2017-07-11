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
#include "../C/activation.h"
#include "primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Activation using rectified linear unit or parameterized rectified linear unit.
/// @details Can get one negative slope or negative slope per channel.
/// @par Algorithm:
///   out(i,x,y) = max(0, in(i,x,y)) + slope(i) * min(0, in(i,x,y))
/// @par Where:
///   @li out(i,x,y) : value at x, y from i-th feature map after activation.
///   @li in(i,x,y) : value at x, y from i-th feature map before activation.
///   @li slope(i) : the slope value of the i-th feature map (can be shared across channels or one slope per channel).
struct activation : public primitive_base<activation, CLDNN_PRIMITIVE_DESC(activation)>
{
    CLDNN_DECLATE_PRIMITIVE(activation)

    /// @brief Constructs Relu primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param activation_func activation function.
    /// @param additional_params additional params (slope/max_val/linear a,b).
    activation(
        const primitive_id& id,
        const primitive_id& input,
        cldnn_activation_func activation_func,
        cldnn_activation_additional_params additional_params = { 0.f,0.f },
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input}, output_padding)
        , activation_func(activation_func)
        , additional_params(additional_params)
        , negative_slope_input("")
    {
    }

    /// @brief Constructs Parameterized Relu primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param slope_input  PRelu activation slopes input primitive id.
    /// Input x dimension should be equal to input feature size (one slope per channel).
    /// All other dimensions should be 1.
    activation(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& slope_input,
        const padding& output_padding = padding()
    )
        : primitive_base(id, { input }, output_padding)
        , activation_func(activation_prelu)
        , additional_params({ 0,0 })
        , negative_slope_input(slope_input)
    {
    }

    /// @brief Constructs a copy from basic C API @CLDNN_PRIMITIVE_DESC{activation}
    activation(const dto* dto)
        : primitive_base(dto)
        , activation_func(dto->activation_func)
        , additional_params(dto->additional_params)
        , negative_slope_input(dto->negative_slope_input)
    {
    }

    /// @brief activation function.
    cldnn_activation_func activation_func;

    /// @brief activation additional params.
    cldnn_activation_additional_params additional_params;

    /// @brief PRelu activation slope input primitive id.
    /// Input x dimension should be equal to input feature size (one slope per channel).
    /// All other dimensions should be 1.
    primitive_id negative_slope_input;

protected:

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override
    {
        if (negative_slope_input.empty())
            return{};
        return{ negative_slope_input };
    }

    void update_dto(dto& dto) const override
    {
        dto.activation_func = activation_func;
        dto.additional_params = additional_params;
        dto.negative_slope_input = negative_slope_input.c_str();
    }
};
/// @}
/// @}
/// @}
}