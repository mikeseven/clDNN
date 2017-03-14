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
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after reorder.
/// Also merged with subtraction layer, which can subtract values while doing reordering.
/// NOTE THAT THIS WILL SUBTRACT THE SAME VALUES FROM EACH BATCH.
struct reorder : public primitive_base<reorder, CLDNN_PRIMITIVE_DESC(reorder)>
{
    CLDNN_DECLATE_PRIMITIVE(reorder)

    /// @brief Constructs reorder primitive with directly provided mean subtract values.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param values_to_substract Array of mean subtract values.
    reorder(
        const primitive_id& id,
        const primitive_id& input,
        const layout& output_layout,
        const std::vector<float>& values_to_substract = {},
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        : primitive_base(id, { input }, input_padding, output_padding)
        , output_layout(output_layout)
        , mean("")
        , substract_per_feature(values_to_substract)
    {
    }

    /// @brief Constructs reorder primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    reorder(
        const primitive_id& id,
        const primitive_id& input,
        const layout& output_layout,
        primitive_id mean,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        : primitive_base(id, { input }, input_padding, output_padding)
        , output_layout(output_layout)
        , mean(mean)
        , substract_per_feature(0)
    {
    }

    /// @brief Constructs a copy from basic C API @CLDNN_PRIMITIVE_DESC{reorder}
    reorder(const dto* dto)
        : primitive_base(dto)
        , output_layout(dto->output_layout)
        , mean(dto->mean_substract)
        , substract_per_feature(float_arr_to_vector(dto->substract_per_feature))
    {
    }

    /// @brief Requested memory layout.
    layout output_layout;
    /// @brief Primitive id to get mean subtract values. Ignored if substract_per_featrue is set.
    primitive_id mean;
    /// @brief Array of mean subtract values.
    std::vector<float> substract_per_feature;

protected:
    std::vector<primitive_id> get_dependencies() const override 
    {
        if (mean.empty())
            return{};
        return{ mean };
    }

    void update_dto(dto& dto) const override
    {
        dto.output_layout = output_layout;
        dto.mean_substract = mean.c_str();
        dto.substract_per_feature = float_vector_to_arr(substract_per_feature);
    }
};
/// @}
/// @}
/// @}
}
