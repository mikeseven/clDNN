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
enum class pooling_mode {max, average};

BEGIN_DTO(pooling)
    pooling_mode mode;
    tensor stride;
    tensor size;
END_DTO(pooling)

struct pooling : public primitive_base<pooling, DTO(pooling)>
{
    DLL_SYM static primitive_type_id type_id();
    typedef DTO(pooling) dto;

    pooling(
        const primitive_id& id,
        const primitive_id& input,
        pooling_mode mode,
        const tensor& stride,
        const tensor& size,
        const tensor& input_offset = { format::x,0,{ 0 } },
        const tensor& output_offset = { format::x,0,{ 0 } },
        const padding_types padding_type = padding_types::zero
        )
        :primitive_base(id, {input}, input_offset, output_offset, padding_type, mode, stride, size)
    {}

    pooling(const dto* dto)
        :primitive_base(dto)
    {}

    pooling_mode mode() const { return _dto.mode; }
    tensor stride() const { return _dto.stride; }
    tensor size() const { return _dto.size; }
};

}