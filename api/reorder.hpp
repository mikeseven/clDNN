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
#include "primitive.hpp"

namespace cldnn
{

BEGIN_DTO(reorder)
    format::type output_format;
    primitive_id_ref mean_substract;
END_DTO(reorder)


class reorder : public primitive_base<reorder, DTO(reorder)>
{
public:
    typedef DTO(reorder) dto;
    DLL_SYM static primitive_type type_id();

    explicit reorder(
        const primitive_id& id,
        const primitive_id& input,
        format ofm,
        primitive_id mean = "",
        const tensor& input_offset = { format::x,0,{ 0 } },
        const tensor& output_offset = { format::x,0,{ 0 } },
        const padding_types padding_type = padding_types::zero
    )
        : primitive_base(id, { input }, input_offset, output_offset, padding_type, ofm)
    {
        // use the same storage for input and mean_substract
        _input.push_back(mean);
        _dto.mean_substract = _input.store().back();
    }

    explicit reorder(const dto* dto)
        :primitive_base(dto)
    {
        // use the same storage for input and mean_substract
        _input.push_back(dto->mean_substract);
        _dto.mean_substract = _input.store().back();
    }

    format output_format() const { return _dto.output_format; }
    primitive_id mean_substract() const { return _dto.mean_substract; }
};
}