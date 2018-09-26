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
#include "../C/gemm.h"
#include "primitive.hpp"

namespace cldnn
{
    /// @addtogroup cpp_api C++ API
    /// @{
    /// @addtogroup cpp_topology Network Topology
    /// @{
    /// @addtogroup cpp_primitives Primitives
    /// @{
    /// @brief Type of gemm that will be added to the input by border layer / primitive.

    /// @brief Adds gemm  input.
    ///
    /// @details General Matrix Multiplication witch batch support, 
    ///          A(B,Z,X)xA2(B,Y,Z)=C(B,X,Y)
    /// @n
    /// @n@b Requirements:
    /// @n - @c input - first matrix
    /// @n - @c input2 - second matrix
    /// @n - @c optional: out_bias matrix, alpha, beta, transpose
    /// @n - @c computations with optional params: output = alpha x (outbias x beta + input x input2) 
    /// @n - @c transpose params tranposing second matrix <-TODO


struct gemm : public primitive_base<gemm, CLDNN_PRIMITIVE_DESC(gemm)>
{
    CLDNN_DECLARE_PRIMITIVE(gemm)

        /// @brief Constructs lstm layer.
        /// @param id This primitive id.
        /// @param input Vector of primitive id.

        gemm(
            const primitive_id& id,
            const primitive_id& input,
            const primitive_id& input2,
            const bool transpose_input1 = false,
            const bool transpose_input2 = false,
            const float alpha = 1.0f,
            const float beta = 0.0f,
            const padding& output_padding = padding()
        )
        : primitive_base(id, { input, input2 }, output_padding)
        , transpose_input1(transpose_input1)
        , transpose_input2(transpose_input2)
        , alpha(alpha)
        , beta(beta)
    {
    }

        gemm(
            const primitive_id& id,
            const primitive_id& input,
            const primitive_id& input2,
            const primitive_id& out_bias,
            const float alpha,
            const float beta,
            const padding& output_padding = padding()
        )
        : primitive_base(id, { input, input2, out_bias }, output_padding)
        , alpha(alpha)
        , beta(beta)

    {
    }

        gemm(
            const primitive_id& id,
            const primitive_id& input,
            const primitive_id& input2,
            const primitive_id& out_bias,
            const float alpha,
            const float beta,
            const bool transpose_input1 = false,
            const bool transpose_input2 = false,
            const padding& output_padding = padding()
        )
        : primitive_base(id, { input, input2, out_bias }, output_padding)
        , alpha(alpha)
        , beta(beta)
        , transpose_input1(transpose_input1)
        , transpose_input2(transpose_input2)

    {
    }


    float alpha;
    float beta;
    bool transpose_input1;
    bool transpose_input2;

    /// @brief Constructs a copy from basic C API @CLDNN_PRIMITIVE_DESC{gemm}

    gemm(const dto* dto)
        : primitive_base(dto)
        , transpose_input1 (dto->transpose_input1)
        , transpose_input2(dto->transpose_input2)
        , alpha (dto->alpha)
        , beta (dto->beta)
    {
    }

    void update_dto(dto& dto) const override
    {
        dto.alpha = alpha;
        dto.beta = beta;
        dto.transpose_input1 = transpose_input1;
        dto.transpose_input2 = transpose_input2;
    }
};

}

/// @}
/// @}
/// @}
