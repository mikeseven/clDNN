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
#include "api/primitives/crop.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
class typed_program_node<crop> : public typed_program_node_base<crop>
{
public:
    auto& input() const { return get_dependency(0); }
    auto& reference_input() const { return get_dependency(1); }
};

using crop_node = typed_program_node<crop>;

template <>
class typed_primitive_inst<crop> : public typed_primitive_inst_base<crop>
{
    using parent = typed_primitive_inst_base<crop>;

public:
    static layout calc_output_layout(crop_node const& node);

public:
    typed_primitive_inst(network_impl& network, crop_node const& node);

    const memory& input_memory() const { return dep_memory(0); }
    const memory& reference_input_memory() const { return dep_memory(1); }
};

using crop_inst = typed_primitive_inst<crop>;
}
