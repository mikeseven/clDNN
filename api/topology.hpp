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
struct topology
{
    topology()
        : _impl(check_status<cldnn_topology>("failed to create topology", cldnn_create_topology))
    {}

    template<class ...Args>
    topology(const Args&... args)
        : topology()
    {
        add<Args...>(args...);
    }

    topology(const topology& other) :_impl(other._impl)
    {
        retain();
    }

    topology& operator=(const topology& other)
    {
        if (_impl == other._impl) return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    ~topology()
    {
        release();
    }

    friend bool operator==(const topology& lhs, const topology& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const topology& lhs, const topology& rhs) { return !(lhs == rhs); }

    template<class PType>
    void add(const PType& desc)
    {
        check_status<void>("primitive add failed", [&](status_t* status) { cldnn_add_primitive(_impl, desc.get_dto(), status); });
    }

    template<class PType, class ...Args>
    void add(const PType& desc, Args... args)
    {
        check_status<void>("primitive add failed", [&](status_t* status) { cldnn_add_primitive(_impl, desc.get_dto(), status); });
        add<Args...>(args...);
    }

    cldnn_topology get() const { return _impl; }

private:
    friend struct engine;
    friend struct network;
    cldnn_topology _impl;
    topology(cldnn_topology impl) :_impl(impl)
    {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }

    void retain()
    {
        check_status<void>("retain topology failed", [=](status_t* status) { cldnn_retain_topology(_impl, status); });
    }
    void release()
    {
        check_status<void>("retain topology failed", [=](status_t* status) { cldnn_release_topology(_impl, status); });
    }
};

CLDNN_API_CLASS(topology)

}