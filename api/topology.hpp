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
typedef struct topology_impl* cldnn_topology_t;
struct topology
{
    topology()
        : _impl(check_status<cldnn_topology_t>("failed to create topology", create_topology_impl))
    {}

    template<class ...Args>
    topology(const Args&... args)
        : topology()
    {
        add<Args...>(args...);
    }

    topology(const topology& other) :_impl(other._impl)
    {
        retain_topology(_impl);
    }

    topology& operator=(const topology& other)
    {
        if (_impl == other._impl) return *this;
        release_topology(_impl);
        _impl = other._impl;
        retain_topology(_impl);
        return *this;
    }

    ~topology()
    {
        release_topology(_impl);
    }

    friend bool operator==(const topology& lhs, const topology& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const topology& lhs, const topology& rhs) { return !(lhs == rhs); }

    template<class PType>
    void add(const PType& desc)
    {
        status_t status = add_primitive_dto(_impl, desc.get_dto());
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("primitive add failed", status);
    }

    template<class PType, class ...Args>
    void add(const PType& desc, Args... args)
    {
        status_t status = add_primitive_dto(_impl, desc.get_dto());
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("primitive add failed", status);
        add<Args...>(args...);
    }

    topology_impl* get() const { return _impl; }

private:
    friend struct engine;
    friend struct network;
    cldnn_topology_t _impl;
    topology(cldnn_topology_t impl) :_impl(impl)
    {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }

    DLL_SYM static cldnn_topology_t create_topology_impl(status_t* status);
    DLL_SYM static status_t add_primitive_dto(cldnn_topology_t topology, const primitive_dto* dto);
    DLL_SYM static void retain_topology(cldnn_topology_t topology);
    DLL_SYM static void release_topology(cldnn_topology_t topology);
};

API_CLASS(topology)

}