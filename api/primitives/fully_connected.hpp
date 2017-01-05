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
BEGIN_DTO(fully_connected)
    uint32_t with_activation;
    float activation_negative_slope;
    primitive_id_ref weights;
    primitive_id_ref bias;
END_DTO(fully_connected)

struct fully_connected : public primitive_base<fully_connected, DTO(fully_connected)>
{
    DLL_SYM static primitive_type_id type_id();
    typedef DTO(fully_connected) dto;
    
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
        , with_activation(_dto.with_activation)
        , negative_slope(_dto.activation_negative_slope)
    {
        init_dto();
    }

    fully_connected(const dto* dto)
        :primitive_base(dto)
        , weights(dto->weights)
        , bias(dto->bias)
        , with_activation(_dto.with_activation)
        , negative_slope(_dto.activation_negative_slope)
    {
        init_dto();
    }

    const primitive_id weights;
    const primitive_id bias;
    const uint32_t& with_activation;
    const float& negative_slope;
protected:
    std::vector<primitive_id> get_dependencies() const override { return{ weights, bias }; }

private:
    void init_dto()
    {
        _dto.weights = weights;
        _dto.bias = bias;
    }
};
}