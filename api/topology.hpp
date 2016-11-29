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
#include <cstdint>
#include "cldnn_defs.h"
#include "compounds.h"
#include "memory.hpp"
#include "primitive.hpp"

namespace cldnn {

struct context;
class topology_impl;

struct topology
{
    typedef topology_impl impl_type;
    DLL_SYM topology(const topology& other);
    DLL_SYM topology& operator=(const topology& other);
    DLL_SYM ~topology();
    friend bool operator==(const topology& lhs, const topology& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const topology& lhs, const topology& rhs) { return !(lhs == rhs); }

    void add_data(primitive_id id, const memory& mem)
    {
        add_primitive(data(id, mem));
    }

    void add_input(primitive_id id, const layout& layout)
    {
        add_primitive(input_layout(id, layout));
    }

    template<class PType>
    void add_primitive(PType&& desc)
    {
        status_t status = add_primitive_dto(desc.get_dto());
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("primitive add failed", status);
    }

    DLL_SYM context get_context() const;
    topology_impl* implementation() const { return _impl; }
private:
    friend struct context;
    friend struct engine;
    topology_impl* _impl;
    topology(topology_impl* impl) :_impl(impl) {}

    DLL_SYM status_t add_primitive_dto(const primitive_dto* dto);
};

static_assert(std::is_standard_layout<topology>::value, "class has to be 'standart layout'");

}