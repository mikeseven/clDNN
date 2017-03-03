/*
// Copyright (c) 2017 Intel Corporation
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
#include "roi_pooling.h"
#include "../primitive.hpp"

namespace cldnn
{


struct roi_pooling : public primitive_base<roi_pooling, CLDNN_PRIMITIVE_DESC(roi_pooling)>
{
    CLDNN_DECLATE_PRIMITIVE(roi_pooling)

    roi_pooling(
        const primitive_id& id,
        const primitive_id& input_data,
        const primitive_id& input_rois,
        int pooled_width,
        int pooled_height,
        float spatial_scale,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input_data, input_rois}, input_padding, output_padding)
        , pooled_width(pooled_width)
        , pooled_height(pooled_height)
        , spatial_scale(spatial_scale)
    {}

    roi_pooling(const dto* dto)
        : primitive_base(dto)
        , pooled_width(dto->pooled_width)
        , pooled_height(dto->pooled_height)
        , spatial_scale(dto->spatial_scale)
    {}

    int pooled_width;
    int pooled_height;
    float spatial_scale;

protected:
    void update_dto(dto& dto) const override
    {
        primitive_base::update_dto(dto);
        dto.pooled_width = pooled_width;
        dto.pooled_height = pooled_height;
        dto.spatial_scale = spatial_scale;
    }
};

}