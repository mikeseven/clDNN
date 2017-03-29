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
#include "api/primitives/data.hpp"
#include "primitive_inst.h"
#include "topology_impl.h"

namespace cldnn
{

template <>
class typed_primitive_inst<data> : public typed_primitive_inst_base<data>
{
    using parent = typed_primitive_inst_base<data>;

public:
    static layout calc_output_layout(const topology_map&, std::shared_ptr<const data> desc)
    {
        return desc->mem.get_layout();
    }

public:
    typed_primitive_inst(network_impl& network, std::shared_ptr<const data> desc);
};

using data_inst = typed_primitive_inst<data>;

}
