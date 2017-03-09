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
#include "softmax.h"
#include "../primitive.hpp"

namespace cldnn
{
struct softmax : public primitive_base<softmax, CLDNN_PRIMITIVE_DESC(softmax)>
{
    CLDNN_DECLATE_PRIMITIVE(softmax)

    softmax(
        const primitive_id& id,
        const primitive_id& input,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :primitive_base(id, {input}, input_padding, output_padding)
    {}

    softmax(const dto* dto)
        :primitive_base(dto)
    {}

private:
    void update_dto(dto&) const override {}
};
}