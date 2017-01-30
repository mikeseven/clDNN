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
#include "fully_connected.h"
#include "../primitive.hpp"

namespace cldnn
{
struct fully_connected : public primitive_base<fully_connected, CLDNN_PRIMITIVE_DESC(fully_connected)>
{
    CLDNN_DECLATE_PRIMITIVE(fully_connected)

    fully_connected(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& weights,
        const primitive_id& bias,
        bool with_activation = false,
        float activation_slp = 0.0f,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input}, input_padding, output_padding, static_cast<uint32_t>(with_activation), activation_slp, "", "")
        , weights(weights)
        , bias(bias)
        , with_activation(with_activation)
        , activation_negative_slope(_dto.activation_negative_slope)
    {
        init_dto();
    }

    fully_connected(const dto* dto)
        :primitive_base(dto)
        , weights(dto->weights)
        , bias(dto->bias)
        , with_activation(_dto.with_activation != 0)
        , activation_negative_slope(_dto.activation_negative_slope)
    {
        init_dto();
    }

    const primitive_id weights;
    const primitive_id bias;
    const bool with_activation;
    const float activation_negative_slope;
protected:
    std::vector<primitive_id> get_dependencies() const override { return{ weights, bias }; }

private:
    void init_dto()
    {
        _dto.weights = weights.c_str();
        _dto.bias = bias.c_str();
    }
};
}