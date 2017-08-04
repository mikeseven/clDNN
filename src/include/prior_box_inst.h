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
#include "api/CPP/prior_box.hpp"
#include "primitive_inst.h"

#include <boost/optional.hpp>

namespace cldnn
{

using prior_box_node = typed_program_node<prior_box>;

template <>
class typed_primitive_inst<prior_box> : public typed_primitive_inst_base<prior_box>
{
    using parent = typed_primitive_inst_base<prior_box>;

public:
    static layout calc_output_layout(prior_box_node const& node);
    static std::string to_string(prior_box_node const& node);
public:
    typed_primitive_inst(network_impl& network, prior_box_node const& node);

    decltype(auto) input_memory() const { return dep_memory(0); }

private:
    template<typename dtype> void generate_output();
};

using prior_box_inst = typed_primitive_inst<prior_box>;

}
