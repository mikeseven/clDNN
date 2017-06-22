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
#include "../C/pooling.h"
#include "primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Select method for the @ref pooling layer.
enum class pooling_mode : int32_t
{
    /// @brief Maximum-pooling method.
    max     = cldnn_pooling_max,
    /// @brief Average-pooling method.
    average = cldnn_pooling_average
};

/// @brief Performs "pooling" operation which is a form of non-linear down-sampling.
/// @details Pools the input image by taking the max, average, etc. within regions.
struct pooling : public primitive_base<pooling, CLDNN_PRIMITIVE_DESC(pooling)>
{
    CLDNN_DECLATE_PRIMITIVE(pooling)

    /// @brief Constructs pooling primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param mode Pooling mode.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param size Pooling kernel size.
    pooling(
        const primitive_id& id,
        const primitive_id& input,
        pooling_mode mode,
        const tensor& size,
        const tensor& stride,
        const tensor& input_offset = { 0,0,0,0 },
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input}, output_padding)
        , mode(static_cast<pooling_mode>(mode))
        , input_offset(input_offset)
        , stride(stride)
        , size(size)
        , with_output_size(false)
    {}

    /// @brief Constructs pooling primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param mode Pooling mode.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param size Pooling kernel size.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    pooling(
        const primitive_id& id,
        const primitive_id& input,
        pooling_mode mode,
        const tensor& size,
        const tensor& stride,
        const tensor& input_offset,
        tensor output_size,
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input}, output_padding)
        , mode(static_cast<pooling_mode>(mode))
        , input_offset(input_offset)
        , stride(stride)
        , size(size)
        , with_output_size(true)
        , output_size(output_size)
    {}

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{pooling}
    pooling(const dto* dto)
        : primitive_base(dto)
        , mode(static_cast<pooling_mode>(dto->mode))
        , input_offset(dto->input_offset)
        , stride(dto->stride)
        , size(dto->size)
        , with_output_size(dto->with_output_size != 0)
        , output_size(dto->output_size)
    {}

    /// @brief Constructs pooling primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param mode Pooling mode.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param size Pooling kernel size.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    /// @return Pooling primitive with specified settings.
    static pooling create_with_output_size(
        const primitive_id& id,
        const primitive_id& input,
        tensor output_size,
        pooling_mode mode,
        const tensor& size,
        const tensor& stride,
        const tensor& input_offset = { 0,0,0,0 },
        const padding& output_padding = padding()
    )
    {
        return pooling(id, input, mode, size, stride, input_offset, output_size, output_padding);
    }

    /// @brief Pooling mode.
    pooling_mode mode;
    /// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the pooling window should start calculations.
    tensor input_offset;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    tensor stride;
    /// @brief Pooling kernel size.
    tensor size;
    /// @brief Indicates that the primitive has user-defined output size (non-zero value).
    bool with_output_size;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;

protected:
    void update_dto(dto& dto) const override
    {
        dto.mode = static_cast<int32_t>(mode);
        dto.input_offset = input_offset;
        dto.stride = stride;
        dto.size = size;
        dto.with_output_size = with_output_size;
        dto.output_size = output_size;
    }
};
/// @}
/// @}
/// @}
}