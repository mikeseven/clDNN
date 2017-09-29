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
#include "../C/assign_patch.h"
#include "primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs elementwise operations (sum, subtract, max or product) on two input primitives
/// Also supports built-in Relu @ref activation available by setting it in arguments.
/// @notes
/// - both inputs have to have equal sizes in all dimensions
/// - format of both inputs has to be the same
struct assign_patch : public primitive_base<assign_patch, CLDNN_PRIMITIVE_DESC(assign_patch)>
{
    CLDNN_DECLATE_PRIMITIVE(assign_patch)

    /// @brief Constructs assign_patch primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param input2 Second input primitive id with values needed for assign_patch computation.
    /// @param mode assign_patch mode.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    assign_patch(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& nn,
        bool with_activation = false,
        float activation_slp = 0.0f,
        const padding& output_padding = padding()
    )
        :primitive_base(id, { input, nn }, output_padding)
        , with_activation(with_activation)
        , activation_negative_slope(activation_slp)
    {
    }

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{assign_patch}
    assign_patch(const dto* dto)
        :primitive_base(dto)
        , with_activation(dto->with_activation != 0)
        , activation_negative_slope(dto->activation_negative_slope)
    {
        if (dto->input.size != 2)
            throw std::invalid_argument("assign_patch dto should containt exactly two inputs");
    }

    /// @brief Enables Relu activation.
    bool with_activation;
    /// @brief Relu activation slope.
    float activation_negative_slope;

protected:
    void update_dto(dto& dto) const override
    {
        dto.with_activation = with_activation;
        dto.activation_negative_slope = activation_negative_slope;
    }
};
/// @}
/// @}
/// @}
}
