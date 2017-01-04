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
#include "primitive.hpp"

namespace cldnn {
class topology_impl;
struct topology
{
    topology()
        : _impl(check_status<topology_impl*>("failed to create topology", create_topology_impl))
    {}

    template<class ...Args>
    topology(const Args&... args)
        : topology()
    {
        add<Args...>(args...);
    }

    DLL_SYM topology(const topology& other);
    DLL_SYM topology& operator=(const topology& other);
    DLL_SYM ~topology();
    friend bool operator==(const topology& lhs, const topology& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const topology& lhs, const topology& rhs) { return !(lhs == rhs); }

    template<class PType>
    void add(const PType& desc)
    {
        status_t status = add_primitive_dto(desc.get_dto());
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("primitive add failed", status);
    }

    template<class PType, class ...Args>
    void add(const PType& desc, Args... args)
    {
        status_t status = add_primitive_dto(desc.get_dto());
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("primitive add failed", status);
        add<Args...>(args...);
    }

    topology_impl* get() const { return _impl; }

private:
    friend struct engine;
    friend struct network;
    topology_impl* _impl;
    topology(topology_impl* impl) :_impl(impl) {}

    DLL_SYM static topology_impl* create_topology_impl(status_t* status);
    DLL_SYM status_t add_primitive_dto(const primitive_dto* dto);
};

API_CLASS(topology)

}