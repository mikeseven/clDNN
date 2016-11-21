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
#include <cstdint>
#include "cldnn_defs.h"
#include "compounds.h"
#include "memory.hpp"
#include "primitive.hpp"

namespace cldnn
{
enum class pooling_mode { max, average };

struct mean_substract_desc : primitive_desc_base<primitive_types::mean_substract>
{
    primitive_id mean;
    mean_substract_desc(primitive_id input, primitive_id mean)
        : primitive_desc_base(input), mean(mean)
    {}
};


struct activation_desc : primitive_desc_base<primitive_types::activation>
{
    float negative_slope;
    activation_desc(primitive_id input, float slp = 0.0f)
        : primitive_desc_base(input)
        , negative_slope(slp)
    {}
};


struct fully_connected_desc : primitive_desc_base<primitive_types::fully_connected>
{
    array_ref<primitive_id> weights;
    array_ref<primitive_id> bias;
    bool with_activation;
    float activation_negative_slope;

    fully_connected_desc(primitive_id input,
        array_ref<primitive_id> weights,
        array_ref<primitive_id>bias,
        bool with_activation = false,
        float activation_slp = 0.0f
    )
        :primitive_desc_base(input)
        , weights(weights)
        , bias(bias)
        , with_activation(with_activation)
        , activation_negative_slope(activation_slp)
    {
        if (weights.size() != bias.size()) throw std::logic_error("weights and bias numbers do not match");
    }
};

struct pooling_desc : primitive_desc_base<primitive_types::pooling>
{
    pooling_mode mode;
    tensor stride;
    tensor size;
    pooling_desc(primitive_id input, pooling_mode mode, tensor stride, tensor size)
        :primitive_desc_base(input), mode(mode), stride(stride), size(size) {}
};


struct normalization_desc : primitive_desc_base<primitive_types::normalization>
{
    uint32_t size;
    float k;
    float alpha;
    float beta;

    normalization_desc(primitive_id input, uint32_t size, float k, float alpha, float beta)
        :primitive_desc_base(input), size(size), k(k), alpha(alpha), beta(beta)
    {}
};


struct softmax_desc : primitive_desc_base<primitive_types::softmax>
{
    softmax_desc(primitive_id input)
        :primitive_desc_base(input)
    {}
};

struct depth_concat_desc : primitive_desc_base<primitive_types::depth_concat>
{

    depth_concat_desc(array_ref<primitive_id> inputs)
        :primitive_desc_base(inputs)
    {}
};

template<> struct primitive_type_traits<primitive_types::reorder>        { typedef reorder_desc primitive_type;         };
template<> struct primitive_type_traits<primitive_types::mean_substract> { typedef mean_substract_desc primitive_type;  };
template<> struct primitive_type_traits<primitive_types::activation>     { typedef activation_desc primitive_type;      };
template<> struct primitive_type_traits<primitive_types::convolution>    { typedef convolution_desc primitive_type;     };
template<> struct primitive_type_traits<primitive_types::fully_connected>{ typedef fully_connected_desc primitive_type; };
template<> struct primitive_type_traits<primitive_types::pooling>        { typedef pooling_desc primitive_type;         };
template<> struct primitive_type_traits<primitive_types::normalization>  { typedef normalization_desc primitive_type;   };
template<> struct primitive_type_traits<primitive_types::softmax>        { typedef softmax_desc primitive_type;         };
template<> struct primitive_type_traits<primitive_types::depth_concat>   { typedef depth_concat_desc primitive_type;    };

}
