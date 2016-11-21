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
#include "tensor.hpp"
#include "primitive.hpp"

namespace cldnn
{

BEGIN_DTO(convolution)
    array_ref<primitive_id_ref> weigths;
    array_ref<primitive_id_ref> bias;
    tensor stride;
    uint32_t with_activation;
    float activation_negative_slope;
END_DTO(convolution)

BEGIN_DESC(convolution)
public:
    convolution_desc(const primitive_dto* dto)
        :primitive_desc_base(dto)
        , _weights(reinterpret_cast<const convolution_dto*>(dto)->weigths)
        , _bias(reinterpret_cast<const convolution_dto*>(dto)->bias)
    {
        _dto.weigths = _weights;
        _dto.bias = _bias;
    }

    convolution_desc(
        const primitive_id& input,
        const std::vector<primitive_id>& weights,
        const std::vector<primitive_id>& bias,
        tensor input_offset = { format::xy,{ 0, 0 } },
        tensor stride = { format::xy,{ 1, 1 } },
        bool with_activation = false,
        float activation_slp = 0.0f
    )
        :primitive_desc_base({ input }), _weights(weights), _bias(bias)
    {
        _dto.input = _inputs;
        _dto.input_offset = input_offset;
        _dto.weigths = _weights;
        _dto.bias = _bias;
        _dto.stride = stride;
        _dto.with_activation = with_activation;
        _dto.activation_negative_slope = activation_slp;
    }

private:
    primitive_id_arr _weights;
    primitive_id_arr _bias;
END_DESC(convolution)
}