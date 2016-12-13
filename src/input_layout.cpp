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
#include "api/topology.hpp"
#include "topology_impl.h"
#include "primitive_type.h"
#include "api/primitives/input_layout.hpp"
#include "input_layout_arg.h"

namespace cldnn
{
    struct input_layout_type : public primitive_type
    {
        std::shared_ptr<const primitive> from_dto(const primitive_dto* dto) const override
        {
            if (dto->type != this) throw std::invalid_argument("dto: primitive type mismatch");
            return std::make_shared<input_layout>(dto->as<input_layout>());
        }
        std::shared_ptr<const primitive_arg> create_arg(network_impl& network, std::shared_ptr<const primitive> desc) const override
        {
            if (desc->type() != this) throw std::invalid_argument("desc: primitive type mismatch");
            return std::make_shared<input_arg>(network, std::static_pointer_cast<const input_layout>(desc));
        }
    };

    primitive_type_id input_layout::type_id()
    {
        static input_layout_type instance;
        return &instance;
    }
}
