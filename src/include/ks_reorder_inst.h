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
#include "ks_reorder.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<ks_reorder> : public typed_program_node_base<ks_reorder>
{
    using parent = typed_program_node_base<ks_reorder>;
public:
    using parent::parent;

    auto& input() const { return get_dependency(0); }
};

using ks_reorder_node = typed_program_node<ks_reorder>;

template <>
class typed_primitive_inst<ks_reorder> : public typed_primitive_inst_base<ks_reorder>
{
    using parent = typed_primitive_inst_base<ks_reorder>;

public:
    static layout calc_output_layout(ks_reorder_node const& node)
    {
        return node.get_primitive()->output_layout;
    }

    static std::string to_string(ks_reorder_node const& node);

public:
    typed_primitive_inst(network_impl& network, ks_reorder_node const& node);

    const memory& input_memory() const { return dep_memory(0); }
};

using ks_reorder_inst = typed_primitive_inst<ks_reorder>;

}
