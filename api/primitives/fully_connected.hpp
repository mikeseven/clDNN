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
        const tensor& input_offset = { format::x,0,{ 0 } },
        const tensor& output_offset = { format::x,0,{ 0 } },
        const padding_types padding_type = padding_types::zero
        )
        : primitive_base(id, {input}, input_offset, output_offset, padding_type, static_cast<uint32_t>(with_activation), activation_slp)
    {
        _input.push_back(weights);
        _dto.weights = _input.store().back();
        _input.push_back(bias);
        _dto.bias = _input.store().back();
    }

    fully_connected(const dto* dto)
        :primitive_base(dto)
    {
        _input.push_back(dto->weights);
        _dto.weights = _input.store().back();
        _input.push_back(dto->bias);
        _dto.weights = _input.store().back();
    }

    primitive_id weights() const { return _dto.weights; }
    primitive_id bias() const { return _dto.bias; }
    bool with_activation() const { return _dto.with_activation; }
    float negative_slope() const { return _dto.activation_negative_slope; }
};
}