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
#include "depth_concatenate.h"
#include "../primitive.hpp"

namespace cldnn
{
struct depth_concatenate : public primitive_base<depth_concatenate, CLDNN_PRIMITIVE_DESC(depth_concatenate)>
{
    CLDNN_DECLATE_PRIMITIVE(depth_concatenate)

    depth_concatenate(
        const primitive_id& id,
        const std::vector<primitive_id>& input
    )
        :primitive_base(id, input, padding(), padding())
    {}

    depth_concatenate(const dto* dto)
        :primitive_base(dto)
    {}
};
}
