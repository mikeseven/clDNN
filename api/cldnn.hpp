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
#include <functional>

namespace cldnn
{

class event_impl;
struct event
{
    typedef event_impl impl_type;
    DLL_SYM event(const event& other);
    DLL_SYM event& operator=(const event& other);
    DLL_SYM ~event();
    friend bool operator==(const event& lhs, const event& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const event& lhs, const event& rhs) { return !(lhs == rhs); }

    DLL_SYM void wait();
    DLL_SYM void reset();
    DLL_SYM void set();
    typedef void(*event_handler)(void*);
    DLL_SYM void on_event(event_handler handler, void* param);

    event_impl* implementation() const { return _impl; }
private:
    friend struct network;
    event(event_impl* impl):_impl(impl){}
    event_impl* _impl;
};

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

    DLL_SYM const memory& get_output(primitive_id_ref id);

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
        return create_obj<event>("network execute failed", [&](status_t* status) { return execute_impl(dependencies, status); });
    }
    DLL_SYM engine get_engine();

    network_impl* implementation() const { return _impl; }
private:
    friend struct engine;
    network(network_impl* impl):_impl(impl){}
    network_impl* _impl;
    DLL_SYM status_t set_input_data_impl(primitive_id_ref id, memory mem);
    DLL_SYM array_ref<primitive_id_ref> get_primitive_keys_impl(status_t* status);
    DLL_SYM event_impl* execute_impl(array_ref<event> dependencies, status_t* status);
};

enum class engine_types { ocl, jit };

struct engine_configuration
{
    uint32_t enable_profiling;
    uint32_t enable_debugging;
    engine_types engine_type;
    string_ref compiler_options;
    engine_configuration(bool profiling = false, bool debug = false, string_ref options = "", engine_types type = engine_types::ocl)
        :enable_profiling(profiling), enable_debugging(debug), engine_type(type), compiler_options(options){}
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

class context_impl;
struct context
{
    typedef context_impl impl_type;
    DLL_SYM context(const context& other);
    DLL_SYM context& operator=(const context& other);
    DLL_SYM ~context();

    friend bool operator==(const context& lhs, const context& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const context& lhs, const context& rhs) { return !(lhs == rhs); }

    static context create()
    {
        return create_obj<context>("failed to create context", create_context_impl);
    }

    topology create_topology()
    {
        return create_obj<topology>("failed to create topology", [&](status_t* status) { return create_topology_impl(status); });
    }

    engine create_engine(uint32_t engine_num, const engine_configuration& configuration)
    {
        return create_obj<engine>("failed to create engine", [&](status_t* status)
        {
            return create_engine_impl(engine_num, &configuration, status);
        });
    }
    DLL_SYM uint32_t engine_count(engine_types type);
    context_impl* implementation() const { return _impl; }
private:
    context(context_impl* impl):_impl(impl){}
    context_impl* _impl;
    DLL_SYM static context_impl* create_context_impl(status_t* status) noexcept;
    DLL_SYM topology_impl* create_topology_impl(status_t* status) noexcept;
    DLL_SYM engine_impl* create_engine_impl(uint32_t engine_num, const engine_configuration* configuration, status_t* status) noexcept;
};

static_assert(std::is_standard_layout<context>::value, "class has to be 'standart layout'");

}
