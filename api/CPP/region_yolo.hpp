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
#include "../C/region_yolo.h"
#include "primitive.hpp"

namespace cldnn
{
    /// @addtogroup cpp_api C++ API
    /// @{
    /// @addtogroup cpp_topology Network Topology
    /// @{
    /// @addtogroup cpp_primitives Primitives
    /// @{

    /// @brief Normalizes results so they sum to 1.
    /// @details
    /// @par Algorithm:
    /// @par Where:
    struct region_yolo : public primitive_base<region_yolo, CLDNN_PRIMITIVE_DESC(region_yolo)>
    {
        CLDNN_DECLATE_PRIMITIVE(region_yolo)

        /// @brief Constructs region_yolo primitive.
        /// @param id This primitive id.
        /// @param input Input primitive id.
        /// @param dimension Defines a scope of normalization (see #dimension).
        region_yolo(
            const primitive_id& id,
            const primitive_id& input,
            const uint32_t classes,
            const uint32_t num,
            const padding& output_padding = padding()
        )
            :primitive_base(id, { input }, output_padding)
            , classes(classes)
            , num(num)
        {}

        /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{region_yolo}
        region_yolo(const dto* dto)
            :primitive_base(dto)
            , classes(dto->classes)
            , num(dto->num)
        {}

        /// @brief Defines a scope of a region yolo normalization
        /// @details
        /// Specific behaviour is determined by these parameters, as follows:
        uint32_t classes;
        uint32_t num;


    private:
        void update_dto(dto& dto) const override
        {
            dto.classes = classes;
            dto.num = num;
        }
    };
    /// @}
    /// @}
    /// @}
}
#pragma once
