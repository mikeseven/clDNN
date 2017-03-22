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
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs transposed convolution.
/// Also supports built-in Relu @ref activation available by setting it in arguments.
/// @details Deconvolution is similar to convolution layer with the weights flipped on the axis and stride and input padding parameters used in opposite sense as in convolution.
/// Look into docs/size_offset_stride_padding.html for description how size, offsets, stride & padding parameter work.
struct deconvolution : public primitive_base<deconvolution, CLDNN_PRIMITIVE_DESC(deconvolution)>
{
    CLDNN_DECLATE_PRIMITIVE(deconvolution)

    /// @brief Constructs deconvolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param input_padding Input padding/offset.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
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
        :primitive_base(id, { input }, input_padding, output_padding)
        , weights(_weights.cpp_ids)
        , bias(_bias.cpp_ids)
        , stride(stride)
        , with_activation(with_activation)
        , activation_negative_slope(activation_slp)
        , _weights(weights)
        , _bias(bias)
    {
    }

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{deconvolution}
    deconvolution(const dto* dto)
        :primitive_base(dto)
        , weights(_weights.cpp_ids)
        , bias(_bias.cpp_ids)
        , stride(dto->stride)
        , with_activation(dto->with_activation != 0)
        , activation_negative_slope(dto->activation_negative_slope)
        , _weights(dto->weights)
        , _bias(dto->bias)
    {
        if (!dto->split || weights.size() != bias.size() || dto->split != weights.size())
            throw std::invalid_argument("Invalid deconvolution dto: bad split value");
    }

    /// @brief List of primitive ids containing weights data.
    std::vector<primitive_id>& weights;
    /// @brief List of primitive ids containing bias data.
    std::vector<primitive_id>& bias;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    tensor stride;
    /// @brief Enables Relu activation.
    bool with_activation;
    /// @brief Relu activation slope.
    float activation_negative_slope;

    /// @brief On how many cards split the computation to.
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
        dto.weights = _weights.ref();
        dto.bias = _bias.ref();
        dto.split = split();
        dto.stride = stride;
        dto.with_activation = with_activation;
        dto.activation_negative_slope = activation_negative_slope;
    }
};
/// @}
/// @}
/// @}
}