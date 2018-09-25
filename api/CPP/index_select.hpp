// Copyright (c) 2018 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "../C/index_select.h"
#include "primitive.hpp"


namespace cldnn
{

    enum class index_select_axis_name : int32_t
    {
        along_b,
        along_f,
        along_y,
        along_x
    };

    struct index_select : public primitive_base<index_select, CLDNN_PRIMITIVE_DESC(index_select)>
    {
        CLDNN_DECLARE_PRIMITIVE(index_select)

        index_select(
            const primitive_id& id,
            const primitive_id& input,
            const primitive_id& indices,
            index_select_axis_name axis = index_select_axis_name::along_b,
            const padding& output_padding = padding()
        )
            : primitive_base(id, {input, indices}, output_padding)
            , axis(axis)
        {}

        /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{broadcast}
        index_select(const dto* dto)
            : primitive_base(dto)
            , axis(static_cast<index_select_axis_name>(dto->axis))
        {}

        index_select_axis_name axis;

        protected:

            void update_dto(dto& dto) const override
            {
                dto.axis = static_cast<cldnn_index_select_axis>(axis);
            }
};
/// @}
/// @}
/// @}
}
