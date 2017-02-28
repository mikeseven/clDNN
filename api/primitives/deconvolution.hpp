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
#include "deconvolution.h"
#include "../primitive.hpp"

namespace cldnn
{

struct deconvolution : public primitive_base<deconvolution, CLDNN_PRIMITIVE_DESC(deconvolution)>
{
    CLDNN_DECLATE_PRIMITIVE(deconvolution)

    deconvolution(
        const primitive_id& id,
        const primitive_id& input,
        const std::vector<primitive_id>& weights,
        const std::vector<primitive_id>& bias,
        const padding& input_padding = { format::yx, {0,0} },
        tensor stride = { format::yx, 0, { 1, 1 } },
        bool with_activation = false,
        float activation_slp = 0.0f,
        const padding& output_padding = { format::yx,{ 0,0 } }
    )
        :primitive_base(id, { input }, input_padding, output_padding,
                        stride,
                        static_cast<uint32_t>(with_activation),
                        activation_slp,
                        static_cast<int32_t>(weights.size()),
                        cldnn_primitive_id_arr{ nullptr, 0 },
                        cldnn_primitive_id_arr{ nullptr, 0 } )
        , _weights(weights)
        , _bias(bias)
        , weights(_weights)
        , bias(_bias)
        , split(_dto.split)
        , stride(_dto.stride)
        , with_activation(with_activation)
        , activation_negative_slope(_dto.activation_negative_slope)
    {
        init_dto();
    }

    deconvolution(const dto* dto)
        :primitive_base(dto)
        , _weights(dto->weights)
        , _bias(dto->bias)
        , weights(_weights)
        , bias(_bias)
        , split(_dto.split)
        , stride(_dto.stride)
        , with_activation(_dto.with_activation != 0)
        , activation_negative_slope(_dto.activation_negative_slope)
    {
        init_dto();
    }

protected:
    const detail::primitive_id_arr _weights;
    const detail::primitive_id_arr _bias;

public:
    const std::vector<primitive_id>& weights;
    const std::vector<primitive_id>& bias;
    const int32_t split;
    const tensor stride;
    const bool with_activation;
    const float activation_negative_slope;

protected:
    std::vector<primitive_id> get_dependencies() const override
    {
        auto result = weights;
        result.insert(result.end(), bias.begin(), bias.end());
        return result;
    }

    void init_dto()
    {
        if (_weights.size() != _bias.size()) throw std::invalid_argument("numbers of weights and biases do not match");
        _dto.weights = _weights.ref();
        _dto.bias = _bias.ref();
    }
};
}