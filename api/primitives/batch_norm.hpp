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
#include "batch_norm.h"
#include "../primitive.hpp"

namespace cldnn
{

struct batch_norm : public primitive_base<batch_norm, CLDNN_PRIMITIVE_DESC(batch_norm)>
{
    CLDNN_DECLATE_PRIMITIVE(batch_norm)

    batch_norm(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& mean,
        const primitive_id& variance,
        bool use_global_stats,
        float epsilon,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :primitive_base(id, {input}, input_padding, output_padding)
        , mean(mean)
        , variance(variance)
        , use_global_stats(use_global_stats)
        , epsilon(epsilon)
    {
    }

    batch_norm(const dto* dto)
        :primitive_base(dto)
        , mean(dto->mean)
        , variance(dto->variance)
        , use_global_stats(dto->use_global_stats)
        , epsilon(dto->epsilon)
    {
    }

    primitive_id mean;
    primitive_id variance;
    bool use_global_stats;
    float epsilon;

protected:
    std::vector<primitive_id> get_dependencies() const override { return{ mean, variance }; }

    void update_dto(dto& dto) const override
    {
        primitive_base::update_dto(dto);
        dto.mean = mean.c_str();
        dto.variance = variance.c_str();
        dto.use_global_stats = use_global_stats;
        dto.epsilon = epsilon;
    }
};
}
