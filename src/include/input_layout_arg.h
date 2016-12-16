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
#include "primitive_arg.h"
#include "api/primitives/input_layout.hpp"
#include "memory_impl.h"
#include "network_impl.h"

namespace cldnn
{
class input_layout_arg : public primitive_arg_base<input_layout>
{
public:
    input_layout_arg(network_impl& network, std::shared_ptr<const input_layout> desc)
        :primitive_arg_base(network, desc, desc->layout)
    {}

    void set_data(const memory& mem)
    {
        if (mem.get_layout() != _output.get_layout())
            throw std::invalid_argument("data layout does not match");
        auto engine = get_network().get_engine();
        if (mem.get()->is_allocated_by(engine))
        {
            _output = mem;
        }
        else
        {
            pointer<char> src(mem);
            pointer<char> dst(_output);
            assert(src.size() == dst.size());
            std::copy(src.begin(), src.end(), dst.begin());
        }
    }
};
}
