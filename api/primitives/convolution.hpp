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

BEGIN_DTO(convolution)
    tensor stride;
    uint32_t with_activation;
    float activation_negative_slope;
    array_ref<primitive_id_ref> weights;
    array_ref<primitive_id_ref> bias;
END_DTO(convolution)


struct convolution : public primitive_base<convolution, DTO(convolution)>
{
    DLL_SYM static primitive_type_id type_id();
    typedef DTO(convolution) dto;

    convolution(
        const primitive_id& id,
        const primitive_id& input,
        const std::vector<primitive_id>& weights,
        const std::vector<primitive_id>& bias,
        const tensor& input_offset = { format::yx,{ 0, 0 } },
        tensor stride = { format::yx,{ 1, 1 } },
        bool with_activation = false,
        float activation_slp = 0.0f,
        const tensor& output_offset = { format::yx,{ 0, 0 } },
        const padding_types padding_type = padding_types::zero
        )
        :primitive_base(id, { input }, input_offset, output_offset, padding_type, stride, static_cast<uint32_t>(with_activation), activation_slp)
    {
        init_weights<primitive_id>(weights, bias);
    }

    convolution(const dto* dto)
        :primitive_base(dto)
    {
        init_weights(dto->weights, dto->bias);
    }

    std::vector<primitive_id> weights() const
    {
        assert(_input.size() > _split * 2);
        return{ _input.store().end() - _split * 2, _input.store().end() - _split };
    }

    std::vector<primitive_id> bias() const
    {
        assert(_input.size() > _split * 2);
        return{ _input.store().end() - _split, _input.store().end() };
    }

    size_t split() const { return _split; }
    const tensor& stride() const { return _dto.stride; };
    bool with_activation() const { return _dto.with_activation; }
    float negative_slope() const { return _dto.activation_negative_slope; };

private:
    size_t _split;

    template<typename T>
    void init_weights(array_ref<T> weights, array_ref<T> bias)
    {
        if (weights.size() != bias.size()) throw std::invalid_argument("numbers of weights and biases do not match");
        _split = weights.size();
        auto input_size = _input.size();
        std::copy(weights.begin(), weights.end(), std::back_inserter(_input));
        std::copy(bias.begin(), bias.end(), std::back_inserter(_input));
        assert(_input.size() == _split * 2 + 1);
        _dto.weights = array_ref<primitive_id_ref>{ _input.ref().data() + input_size, _split };
        _dto.bias = array_ref<primitive_id_ref>{ _input.ref().data() + input_size + _split, _split };
    }
};
}