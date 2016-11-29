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
#include "network.hpp"

namespace cldnn
{
enum class engine_types { ocl, jit };

struct engine_configuration
{
    uint32_t enable_profiling;
    uint32_t enable_debugging;
    engine_types engine_type;
    string_ref compiler_options;
    engine_configuration(bool profiling = false, bool debug = false, string_ref options = "", engine_types type = engine_types::ocl)
        :enable_profiling(profiling), enable_debugging(debug), engine_type(type), compiler_options(options) {}
};

class engine_impl;
struct engine
{
    typedef engine_impl impl_type;
    memory allocate_memory(layout layout)
    {
        status_t status;
        auto buf = allocate_buffer(layout, &status);
        if (!buf)
            CLDNN_THROW("memory allocation failed", status);
        return memory(layout, buf);
    }

    network build_network(topology topology)
    {
        return create_obj<network>("network build failed", [&](status_t* status) { return build_network_impl(topology, status); });
    }

    DLL_SYM context get_context();
    DLL_SYM engine(const engine& other);
    DLL_SYM engine& operator=(const engine& other);
    DLL_SYM ~engine();
    friend bool operator==(const engine& lhs, const engine& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const engine& lhs, const engine& rhs) { return !(lhs == rhs); }

    engine_impl* implementation() const { return _impl; }
private:
    friend struct context;
    explicit engine(engine_impl* impl) :_impl(impl) {}
    engine_impl* _impl;
    DLL_SYM buffer* allocate_buffer(layout layout, status_t* status) noexcept;
    DLL_SYM network_impl* build_network_impl(topology topology, status_t* status) noexcept;
};
}
