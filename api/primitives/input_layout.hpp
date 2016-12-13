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
#include "../memory.hpp"

namespace cldnn
{
BEGIN_DTO(input_layout)
    layout layout;
END_DTO(input_layout)

class input_layout : public primitive_base<input_layout, DTO(input_layout)>
{
public:
    typedef DTO(input_layout) dto;
    DLL_SYM static primitive_type_id type_id();

    input_layout(const primitive_id& id, const layout& layout)
        :primitive_base(id, {}, { format::x, 0,{ 0 } }, { format::x, 0,{ 0 } }, padding_types::zero, layout)
    {}

    explicit input_layout(const dto* dto)
        :primitive_base(dto)
    {}

    layout layout() const { return _dto.layout; }
};
}
