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
#include "../primitive.hpp"

namespace cldnn
{
BEGIN_DTO(batch_norm)
        primitive_id_ref mean;
        primitive_id_ref variance;
        bool use_global_stats;
        float epsilon;
END_DTO(batch_norm)

struct batch_norm : public primitive_base<batch_norm, DTO(batch_norm)>
{
    DLL_SYM static primitive_type_id type_id();
    typedef DTO(batch_norm) dto;

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
        :primitive_base(id, {input}, input_padding, output_padding, "", "", use_global_stats, epsilon)
        , mean(mean)
        , variance(variance)
        , use_global_stats(_dto.use_global_stats)
        , epsilon(_dto.epsilon)
    {
        init_dto();
    }

    batch_norm(const dto* dto)
        :primitive_base(dto)
        , mean(dto->mean)
        , variance(dto->variance)
        , use_global_stats(_dto.use_global_stats)
        , epsilon(_dto.epsilon)
    {
        init_dto();
    }

    const primitive_id mean;
    const primitive_id variance;
    const bool& use_global_stats;
    const float& epsilon;

protected:
    std::vector<primitive_id> get_dependencies() const override { return{ mean, variance }; }

    void init_dto()
    {
        _dto.mean = mean;
        _dto.variance = variance;
    }
};
}
