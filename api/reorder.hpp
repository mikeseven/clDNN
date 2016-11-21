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
#include "tensor.hpp"
#include "primitive.hpp"

namespace cldnn
{
BEGIN_DTO(reorder)
    format::type output_format;
    primitive_id_ref mean_substract;
END_DTO(reorder)

BEGIN_DESC(reorder)
public:
    explicit reorder_desc(const primitive_dto* dto)
        :primitive_desc_base(dto)
    {
    }

    explicit reorder_desc(const primitive_id& input, format ofm, primitive_id mean = "")
        : primitive_desc_base({input})
    {
        _dto.output_format = ofm;
        _dto.mean_substract = mean;
    }
END_DESC(reorder)
}