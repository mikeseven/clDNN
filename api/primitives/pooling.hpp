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
#include "pooling.h"
#include "../primitive.hpp"

namespace cldnn
{
enum class pooling_mode : int32_t
{
    max     = cldnn_pooling_max,
    average = cldnn_pooling_average
};

struct pooling : public primitive_base<pooling, CLDNN_PRIMITIVE_DESC(pooling)>
{
    CLDNN_DECLATE_PRIMITIVE(pooling)

    pooling(
        const primitive_id& id,
        const primitive_id& input,
        pooling_mode mode,
        const tensor& stride,
        const tensor& size,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input}, input_padding, output_padding)
        , mode(static_cast<pooling_mode>(mode))
        , stride(stride)
        , size(size)
    {}

    pooling(const dto* dto)
        : primitive_base(dto)
        , mode(static_cast<pooling_mode>(dto->mode))
        , stride(dto->stride)
        , size(dto->size)
    {}

    pooling_mode mode;
    tensor stride;
    tensor size;

protected:
    void update_dto(dto& dto) const override
    {
        dto.mode = static_cast<int32_t>(mode);
        dto.stride = stride;
        dto.size = size;
    }
};

}