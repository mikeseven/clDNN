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
#include "api/primitives/mean_substract.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<mean_substract> : public typed_program_node_base<mean_substract>
{
public:
    auto& input() const { return get_dependency(0); }
    auto& mean() const { return get_dependency(1); }
};

using mean_substract_node = typed_program_node<mean_substract>;

template <>
class typed_primitive_inst<mean_substract> : public typed_primitive_inst_base<mean_substract>
{
    using parent = typed_primitive_inst_base<mean_substract>;

public:
    static layout calc_output_layout(mean_substract_node const& node);

public:
    typed_primitive_inst(network_impl& network, mean_substract_node const& node);

    const memory& input_memory() const { return dep_memory(0); }
    const memory& mean_memory() const { return dep_memory(1); }
};

using mean_substract_inst = typed_primitive_inst<mean_substract>;

}