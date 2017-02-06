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
#include "input_layout_arg.h"
#include "primitive_type_base.h"
#include "memory_impl.h"

namespace cldnn
{
primitive_type_id input_layout_type_id()
{
    static primitive_type_base<input_layout, input_layout_arg> instance;
    return &instance;
}

input_layout_arg::input_layout_arg(network_impl& network, std::shared_ptr<const input_layout> desc)
    : primitive_arg_base(network, desc, desc->layout)
{
}

void input_layout_arg::set_data(memory_impl* mem)
{
    if (mem->get_layout() != _output.get_layout())
        throw std::invalid_argument("data layout does not match");
    if (mem->is_allocated_by(get_network().get_engine()))
    {
        _output = memory(api_cast(mem), true);
    }
    else
    {
        pointer<char> src(memory(api_cast(mem), true));
        pointer<char> dst(_output);
        std::copy(src.begin(), src.end(), dst.begin());
    }
}

}
