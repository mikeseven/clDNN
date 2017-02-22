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
#include "normalization.h"
#include "../primitive.hpp"

namespace cldnn
{
struct normalization :public primitive_base<normalization, CLDNN_PRIMITIVE_DESC(normalization)>
{
    CLDNN_DECLATE_PRIMITIVE(normalization)

    normalization(
        const primitive_id& id,
        const primitive_id& input,
        uint32_t size,
        float k,
        float alpha,
        float beta,
        cldnn_lrn_norm_region lrn_norm_region,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input}, input_padding, output_padding)
        , size(size)
        , k(k)
        , alpha(alpha)
        , beta(beta)
        , norm_region(lrn_norm_region)
    {}

    normalization(const dto* dto)
        : primitive_base(dto)
        , size(dto->size)
        , k(dto->k)
        , alpha(dto->alpha)
        , beta(dto->beta)
        , norm_region(dto->norm_region)
    {}

    uint32_t size;
    float k;
    float alpha;
    float beta;
    cldnn_lrn_norm_region norm_region;

protected:
    void update_dto(dto& dto) const override
    {
        primitive_base::update_dto(dto);
        dto.size = size;
        dto.k = k;
        dto.alpha = alpha;
        dto.beta = beta;
        dto.norm_region = norm_region;
    }
};
}