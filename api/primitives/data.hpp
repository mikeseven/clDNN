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
#include "data.h"
#include "../primitive.hpp"
#include "../memory.hpp"

namespace cldnn
{
struct data : public primitive_base<data, CLDNN_PRIMITIVE_DESC(data)>
{
    CLDNN_DECLATE_PRIMITIVE(data)

    data(const primitive_id& id, const memory& mem)
        :primitive_base(id, {}, padding(), padding())
        , mem(mem.get(), true)
    {}

    explicit data(const dto* dto)
        :primitive_base(dto)
        , mem(dto->mem, true)
    {}

    memory mem;

protected:
    void update_dto(dto& dto) const override
    {
        dto.mem = mem.get();
    }
};
}
