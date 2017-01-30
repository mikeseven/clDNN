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
#include "scale.h"
#include "../primitive.hpp"

namespace cldnn
{

struct scale : public primitive_base<scale, CLDNN_PRIMITIVE_DESC(scale)>
{
    CLDNN_DECLATE_PRIMITIVE(scale)

    scale(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& scale_input, //should be bfyx or yxfb, where each dimension can be 1, if all dimensions are 1 then this is scalar
        const bool bias_term,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :primitive_base(id, {input}, input_padding, output_padding, "", bias_term, "")
        , scale_input(scale_input)
        , bias_term(_dto.bias_term)
        , bias("")
    {
        init_dto();
    }

    scale(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& scale_input, //should be bfyx or yxfb, where each dimension can be 1, if all dimensions are 1 then this is scalar
        const bool bias_term,
        const primitive_id& bias, //should be same size as scale_input
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :primitive_base(id, { input }, input_padding, output_padding, "", bias_term, "")
        , scale_input(scale_input)
        , bias_term(_dto.bias_term)
        , bias(bias)
    {
        init_dto();
    }

    scale(const dto* dto)
        :primitive_base(dto)
        , scale_input(dto->scale_input)
        , bias_term(_dto.bias_term)
        , bias(dto->bias)
    {
        init_dto();
    }

    const primitive_id scale_input;
    const bool bias_term;
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
        _dto.scale_input = scale_input.c_str();
        _dto.bias = bias.c_str();
    }
};
}
