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
#include "../C/eltwise.h"
#include "primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Select mode for the @ref eltwise layer.
enum class eltwise_mode : int32_t
{
    /// @brief Eltwise sum.
    sum = cldnn_eltwise_sum,
    /// @brief Eltwise subtract.
    sub = cldnn_eltwise_sub,
    /// @brief Eltwise max.
    max = cldnn_eltwise_max,
    /// @brief Eltwise product (Hamarad).
    prod = cldnn_eltwise_prod,
};

/// @brief Performs elementwise operations (sum, subtract, max or product) on two input primitives
/// Also supports built-in Relu @ref activation available by setting it in arguments.
/// @notes
/// - both inputs have to have equal sizes in all dimensions
/// - format of both inputs has to be the same
struct eltwise : public primitive_base<eltwise, CLDNN_PRIMITIVE_DESC(eltwise)>
{
    CLDNN_DECLATE_PRIMITIVE(eltwise)

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param input2 Second input primitive id with values needed for eltwise computation.
    /// @param mode Eltwise mode.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    eltwise(
        const primitive_id& id,
        const std::vector<primitive_id>& inputs,
        eltwise_mode mode,
        bool with_activation = false,
        float activation_slp = 0.0f,
        const padding& output_padding = padding()
    )
        :primitive_base(id, inputs, output_padding)
        , mode(mode)
        , with_activation(with_activation)
        , activation_negative_slope(activation_slp)
    {
    }

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{eltwise}
    eltwise(const dto* dto)
        :primitive_base(dto)
        , mode(static_cast<eltwise_mode>(dto->mode))
        , with_activation(dto->with_activation != 0)
        , activation_negative_slope(dto->activation_negative_slope)
    {
        if (dto->input.size < 2)
            throw std::invalid_argument("eltiwise dto should containt at least two inputs");
    }

    /// @param mode Eltwise mode.
    eltwise_mode mode;
    /// @brief Enables Relu activation.
    bool with_activation;
    /// @brief Relu activation slope.
    float activation_negative_slope;

protected:
    void update_dto(dto& dto) const override
    {
        dto.mode = static_cast<cldnn_eltwise_mode>(mode);
        dto.with_activation = with_activation;
        dto.activation_negative_slope = activation_negative_slope;
    }
};
/// @}
/// @}
/// @}
}
