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
#include "api/primitives/input_layout.hpp"
#include "primitive_arg.h"
#include <memory>
#include "topology_impl.h"

namespace cldnn
{
struct memory_impl;
class input_layout_arg : public primitive_arg_base<input_layout>
{
public:
    static layout calc_output_layout(const topology_map&, std::shared_ptr<const input_layout> desc)
    {
        return desc->layout;
    }

    input_layout_arg(network_impl& network, std::shared_ptr<const input_layout> desc);
    void set_data(memory_impl* mem);
};
}
