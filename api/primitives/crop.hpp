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
#include "crop.h"
#include "../primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

struct crop : public primitive_base<crop, CLDNN_PRIMITIVE_DESC(crop)>
{
    CLDNN_DECLATE_PRIMITIVE(crop)

    crop(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& reference_input,
        const tensor& offsets,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :primitive_base(id, {input}, input_padding, output_padding)
        , reference_input(reference_input)
        , offsets(offsets)
    {
    }

    crop(const dto* dto)
        :primitive_base(dto)
        , reference_input(dto->reference_input)
        , offsets(dto->offsets)
    {
    }

    primitive_id reference_input;
    tensor offsets;

protected:
    std::vector<primitive_id> get_dependencies() const override { return{ reference_input }; }

    void update_dto(dto& dto) const override
    {
        dto.reference_input = reference_input.c_str();
        dto.offsets = offsets;
    }
};
/// @}
/// @}
/// @}
}
