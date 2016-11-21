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
    topology_impl* _impl;

    DLL_SYM status_t add_data(primitive_id id, const memory& mem);
    DLL_SYM status_t add_input(primitive_id id, const layout& layout);
    void add_primitive(primitive_id id, const primitive_desc& desc)
    {
        status_t err = add_primitive_dto(id, desc.get_dto());
        if (err != CLDNN_SUCCESS) throw std::runtime_error("Primitive add failed");
    }

    template<primitive_types Ptype>
    void add_primitive(primitive_id id, typename primitive_type_traits<Ptype>::primitive_type&& desc)
    {
        add_primitive(id, desc);
    }

    DLL_SYM context get_context() const;

    DLL_SYM topology(const topology& other);
    DLL_SYM topology& operator=(const topology& other);
    DLL_SYM ~topology();
private:
    friend struct context;
    explicit topology(topology_impl* impl) :_impl(impl) {}

    DLL_SYM status_t add_primitive_dto(primitive_id_ref id, const primitive_dto* dto);
};
}