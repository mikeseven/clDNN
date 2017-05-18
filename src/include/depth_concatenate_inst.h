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
#include "api/CPP/depth_concatenate.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<depth_concatenate> : public typed_program_node_base<depth_concatenate>
{
public:
    auto& input(size_t idx) const { return get_dependency(idx); }

    auto inputs_count() const { return desc->input.size(); }
};

using depth_concatenate_node = typed_program_node<depth_concatenate>;

template <>
class typed_primitive_inst<depth_concatenate> : public typed_primitive_inst_base<depth_concatenate>
{
    using parent = typed_primitive_inst_base<depth_concatenate>;

public:
    static layout calc_output_layout(depth_concatenate_node const& node);
    static std::string to_string(depth_concatenate_node const& node);

public:
    typed_primitive_inst(network_impl& network, depth_concatenate_node const& node);

    const memory& input_memory(size_t idx) const { return dep_memory(idx); }
};

using depth_concatenate_inst = typed_primitive_inst<depth_concatenate>;

}