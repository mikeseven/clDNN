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
#include "convolution.h"
#include "../primitive.hpp"

namespace cldnn
{

struct convolution : public primitive_base<convolution, CLDNN_PRIMITIVE_DESC(convolution)>
{
    CLDNN_DECLATE_PRIMITIVE(convolution)

    convolution(
        const primitive_id& id,
        const primitive_id& input,
        const std::vector<primitive_id>& weights,
        const std::vector<primitive_id>& bias,
        const padding& input_padding = { format::yx, { 0,0 } },
        tensor stride = { format::yx, 0, { 1, 1 } },
        bool with_activation = false,
        float activation_slp = 0.0f,
        const padding& output_padding = { format::yx, { 0,0 } }
    )
        :primitive_base(id, { input }, input_padding, output_padding)
        , weights(_weights.cpp_ids)
        , bias(_bias.cpp_ids)
        , stride(stride)
        , with_activation(with_activation)
        , activation_negative_slope(activation_slp)
        , _weights(weights)
        , _bias(bias)
    {
        if (weights.size() != bias.size())
            throw std::runtime_error("convolution's weights/bias count does not match");
    }

    convolution(const dto* dto)
        :primitive_base(dto)
        , weights(_weights.cpp_ids)
        , bias(_bias.cpp_ids)
        , stride(dto->stride)
        , with_activation(dto->with_activation != 0)
        , activation_negative_slope(dto->activation_negative_slope)
        , _weights(dto->weights)
        , _bias(dto->bias)
    {
        if (weights.size() != bias.size() || dto->split != weights.size())
            throw std::runtime_error("Invalid convolution dto: bad split value");
    }

    std::vector<primitive_id>& weights;
    std::vector<primitive_id>& bias;
    tensor stride;
    bool with_activation;
    float activation_negative_slope;

    int32_t split() const { return static_cast<int32_t>(weights.size()); }

protected:
    primitive_id_arr _weights;
    primitive_id_arr _bias;

    std::vector<primitive_id> get_dependencies() const override
    {
        auto result = weights;
        result.insert(result.end(), bias.begin(), bias.end());
        return result;
    }

    void update_dto(dto& dto) const override
    {
        primitive_base::update_dto(dto);
        dto.weights = _weights.ref();
        dto.bias = _bias.ref();
        dto.stride = stride;
        dto.split = split();
        dto.with_activation = with_activation;
        dto.activation_negative_slope = activation_negative_slope;
    }
};
}