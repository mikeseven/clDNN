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
BEGIN_DTO(scale)
        primitive_id_ref scale_input;
        bool bias_term;
        primitive_id_ref bias;
END_DTO(scale)

struct scale : public primitive_base<scale, DTO(scale)>
{
    DLL_SYM static primitive_type_id type_id();
    typedef DTO(scale) dto;

    scale(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& scale_input,
        const bool bias_term,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :primitive_base(id, {input}, input_padding, output_padding, scale_input, bias_term)
        , scale_input(scale_input)
        , bias_term(bias_term)
        , bias("")
    {
        init_dto();
    }

    scale(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& scale_input,
        const bool bias_term,
        const primitive_id& bias,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :primitive_base(id, { input }, input_padding, output_padding, scale_input, bias_term, bias)
        , scale_input(scale_input)
        , bias_term(bias_term)
        , bias(bias)
    {
        init_dto();
    }

    scale(const dto* dto)
        :primitive_base(dto)
        , scale_input(dto->scale_input)
        , bias_term(dto->bias_term)
        , bias(dto->bias)
    {
        init_dto();
    }

    const primitive_id scale_input;
    const bool& bias_term;
    const primitive_id bias;

protected:
    std::vector<primitive_id> get_dependencies() const override
    { 
        if (bias.empty())
            return{ scale_input };
        else
            return{ scale_input, bias };
    }

    void init_dto()
    {
        _dto.scale_input = scale_input;
        _dto.bias = bias;
    }
};
}
