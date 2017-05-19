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
#include "../C/softmax.h"
#include "primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Normalizes results so they sum to 1.
/// @details
/// @par Algorithm:
///   b = e^a/sum(N-1; j=0; e^j)
/// @par Where:
///   @li N : number of values to normalize
///   @li b : value after normalization
///   @li a : value before normalization
struct softmax : public primitive_base<softmax, CLDNN_PRIMITIVE_DESC(softmax)>
{
    CLDNN_DECLATE_PRIMITIVE(softmax)

    enum dimension_t
    {
        normalize_bfyx = cldnn_softmax_normalize_bfyx,
        normalize_fyx = cldnn_softmax_normalize_fyx,
        normalize_x = cldnn_softmax_normalize_x,
        normalize_yx = cldnn_softmax_normalize_yx
    };

    /// @brief Constructs softmax primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param dimension Defines a scope of normalization (see #dimension).
    softmax(
        const primitive_id& id,
        const primitive_id& input,
        const dimension_t dimension = normalize_fyx,
        const padding& output_padding = padding()
    )
        :primitive_base(id, {input}, output_padding)
        , dimension(dimension)
    {}

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{softmax}
    softmax(const dto* dto)
        :primitive_base(dto)
        , dimension(static_cast<dimension_t>(dto->dimension))
    {}

    /// @brief Defines a scope of a single softmax normalization as well as number of normalizations.
    /// @details
    /// Being given a 4-dimensional input, which consists of b,f,y,x dimensions, softmax normalizes n data sets of m elements each.
    /// The values of n and m are calculated from b,f,y,x in such way that as a result whole input is processed. Specific behaviour is
    /// determined by this parameter, as follows:
    /// - when set to @p softmax::normalize_x only x-dimension is used as a number of elements @p m and the number of data sets @p n is calculated as b*f*y (each image row is normalized independently)
    /// - when set to @p softmax::normalize_yx @m is defined as x*y and @n as b*f (each 2d image is normalized independently)
    /// - when set to @P softmax::normalize_fyx @m is defined as f*y*x and @n as b (each 3d image is normalized independently)
    /// - when set to @p softmax::normalize_bfyx the whole input is normalized as a single data set
    dimension_t dimension;

private:
    void update_dto(dto& dto) const override
    {
        dto.dimension = static_cast<cldnn_softmax_dimension>(dimension);
    }
};
/// @}
/// @}
/// @}
}