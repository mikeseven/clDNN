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
#include "api/CPP/reorder.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<reorder> : public typed_program_node_base<reorder>
{
    using parent = typed_program_node_base<reorder>;

public:
    using parent::parent;

    auto& input() const { return get_dependency(0); }
    auto& mean() const { return get_dependency(1); }

    bool has_mean() const { return !typed_desc()->mean.empty(); }
    auto can_be_optimized() const { return optimized; }
    void can_be_optimized(bool opt) { optimized = opt; }

private:
    bool optimized = false;
};

using reorder_node = typed_program_node<reorder>;

template <>
class typed_primitive_inst<reorder> : public typed_primitive_inst_base<reorder>
{
    using parent = typed_primitive_inst_base<reorder>;

public:
    static layout calc_output_layout(reorder_node const& node);
    static std::string to_string(reorder_node const& node);

public:
    typed_primitive_inst(network_impl& network, reorder_node const& node);

    size_t inputs_memory_count() const override { return static_cast<size_t>(1); }
    const memory& mean_memory() const { return dep_memory(1); }

    bool has_mean() const { return !argument.mean.empty(); }
};

using reorder_inst = typed_primitive_inst<reorder>;

}
