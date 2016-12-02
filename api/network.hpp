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
#include "topology.hpp"
#include "event.hpp"

namespace cldnn
{

class network_impl;
struct network
{
    typedef network_impl impl_type;
    DLL_SYM network(const network& other);
    DLL_SYM network& operator=(const network& other);
    DLL_SYM ~network();
    friend bool operator==(const network& lhs, const network& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const network& lhs, const network& rhs) { return !(lhs == rhs); }

    DLL_SYM void set_input_data(primitive_id id, memory mem)
    {
        status_t status = set_input_data_impl(id, mem);
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("set data input failed", status);
    }

    const memory& get_output(primitive_id_ref id)
    {
        status_t status;
        auto mem = get_output(id, &status);
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("memory allocation failed", status);
        return mem;
    }

    std::vector<primitive_id> primitive_keys()
    {
        status_t status;
        auto primitives_ref = get_primitive_keys_impl(&status);
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("failed to get primitive keys", status);
        return primitive_id_arr(primitives_ref).store();
    };

    event execute(const std::vector<event>& dependencies)
    {
        return create_obj<event, event_impl>("network execute failed", [&](status_t* status) { return execute_impl(dependencies, status); });
    }

    DLL_SYM engine get_engine();

    network_impl* implementation() const { return _impl; }
private:
    friend struct engine;
    network(network_impl* impl) :_impl(impl) {}
    network_impl* _impl;
    DLL_SYM status_t set_input_data_impl(primitive_id_ref id, memory mem) noexcept;
    DLL_SYM array_ref<primitive_id_ref> get_primitive_keys_impl(status_t* status) noexcept;
    DLL_SYM event_impl* execute_impl(array_ref<event> dependencies, status_t* status) noexcept;
    DLL_SYM const memory& get_output(primitive_id_ref id, status_t* status) noexcept;
};
API_CLASS(network)
}