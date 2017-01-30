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
#include "reorder.h"
#include "../primitive.hpp"
#include "../memory.hpp"

namespace cldnn
{
struct reorder : public primitive_base<reorder, CLDNN_PRIMITIVE_DESC(reorder)>
{
    CLDNN_DECLATE_PRIMITIVE(reorder)

    reorder(
        const primitive_id& id,
        const primitive_id& input,
        const layout& output_layout,
        const std::vector<float>& values_to_substract = {},
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        : primitive_base(id, { input }, input_padding, output_padding, static_cast<cldnn_layout>(output_layout), "", cldnn_float_arr{nullptr, 0})
        , output_layout(_dto.output_layout)
        , mean("")
        , substract_per_feature(values_to_substract)
    {
        init_dto();
    }

    reorder(
        const primitive_id& id,
        const primitive_id& input,
        const layout& output_layout,
        primitive_id mean,
        const padding& input_padding = { format::yx,{ 0, 0 } },
        const padding& output_padding = { format::yx,{ 0, 0 } }
    )
        : primitive_base(id, { input }, input_padding, output_padding, static_cast<cldnn_layout>(output_layout), "", cldnn_float_arr{ nullptr, 0 })
        , output_layout(_dto.output_layout)
        , mean(mean)
        , substract_per_feature(0)
    {
        init_dto();
    }

    reorder(const dto* dto)
        : primitive_base(dto)
        , output_layout(_dto.output_layout)
        , mean(dto->mean_substract)
        , substract_per_feature(float_arr_to_vector(_dto.substract_per_feature))
    {
        init_dto();
    }

    const layout output_layout;
    const primitive_id mean;
    const std::vector<float> substract_per_feature;

protected:
    std::vector<primitive_id> get_dependencies() const override 
    {
        if (mean.empty())
            return{};
        return{ mean };
    }
private:
    void init_dto()
    {
        _dto.mean_substract = mean.c_str();
        _dto.substract_per_feature = float_vector_to_arr(substract_per_feature);
    }

    static std::vector<float> float_arr_to_vector(const cldnn_float_arr& arr)
    {
        std::vector<float> result(arr.size);
        for (size_t i = 0; i < arr.size; i++)
        {
            result[i] = arr.data[i];
        }
        return result;
    }

    static cldnn_float_arr float_vector_to_arr(const std::vector<float>& stor)
    {
        return { stor.data(), stor.size() };
    }

};
}
