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
#include "eltwise.h"
#include "../primitive.hpp"

namespace cldnn
{
enum class eltwise_mode : int32_t
{
    sum = cldnn_eltwise_sum,
    sub = cldnn_eltwise_sub,
    max = cldnn_eltwise_max,
    prod = cldnn_eltwise_prod,
};


struct eltwise : public primitive_base<eltwise, CLDNN_PRIMITIVE_DESC(eltwise)>
{
    CLDNN_DECLATE_PRIMITIVE(eltwise)

    eltwise(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& input2,
        eltwise_mode mode,
        bool with_activation = false,
        float activation_slp = 0.0f,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :primitive_base(id, { input }, input_padding, output_padding)
        , input2(input2)
        , mode(mode)
        , with_activation(with_activation)
        , activation_negative_slope(activation_slp)
    {
    }

    eltwise(const dto* dto)
        :primitive_base(dto)
        , input2(dto->input2)
        , mode(static_cast<eltwise_mode>(dto->mode))
        , with_activation(dto->with_activation != 0)
        , activation_negative_slope(dto->activation_negative_slope)
    {
    }

    primitive_id input2;
    eltwise_mode mode;

    bool with_activation;
    float activation_negative_slope;

protected:
    std::vector<primitive_id> get_dependencies() const override { return{ input2 }; }

    void update_dto(dto& dto) const override
    {
        dto.input2 = input2.c_str();
        dto.mode = static_cast<cldnn_eltwise_mode>(mode);
        dto.with_activation = with_activation;
        dto.activation_negative_slope = activation_negative_slope;
    }
};
}
