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
#include "primitive_type_base.h"
#include "data_arg.h"
#include "memory_impl.h"

namespace cldnn
{
primitive_type_id data::type_id()
{
    static primitive_type_base<data, data_arg> instance;
    return &instance;
}

namespace {
    memory attach_or_copy_data(network_impl& network, memory mem)
    {
        auto engine = network.get_engine();
        if (mem.get()->is_allocated_by(engine))
        {
            return mem;
        }

        memory result(engine->allocate_buffer(mem.get_layout()));
        pointer<char> src(mem);
        pointer<char> dst(result);
        std::copy(src.begin(), src.end(), dst.begin());
        return result;
    }
}

data_arg::data_arg(network_impl& network, std::shared_ptr<const data> desc)
    : primitive_arg_base(network, desc, attach_or_copy_data(network, desc->mem))
{
}

}
