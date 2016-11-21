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
#include "primitives.hpp"

namespace cldnn
{

class event_impl;
struct event
{
    typedef void(*event_handler)(void*);

    DLL_SYM void wait();
    DLL_SYM void reset();
    DLL_SYM void set();
    DLL_SYM void on_event(event_handler handler, void* param);

    DLL_SYM event(const event& other);
    DLL_SYM event& operator=(const event& other);
private:
    friend struct network;
    event(event_impl* impl):_impl(impl){}
    event_impl* _impl;
};

class network_impl;
struct network
{
    DLL_SYM void set_input_data(primitive_id id, memory mem);
    DLL_SYM const memory& get_output(primitive_id id);
    DLL_SYM array_ref<primitive_id> primitive_keys() const;

    DLL_SYM event execute(array_ref<event> dependencies);
    DLL_SYM engine get_engine();

    DLL_SYM network(const network& other);
    DLL_SYM network& operator=(const network& other);
private:
    friend struct engine;
    network(network_impl* impl):_impl(impl){}
    network_impl* _impl;
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
    engine_impl* _impl;

    DLL_SYM memory allocate_memory(layout layout);
    DLL_SYM network build_network(topology topology);
    DLL_SYM context get_context();
    DLL_SYM engine(const engine& other);
    DLL_SYM engine& operator=(const engine& other);
private:
    friend struct context;
    explicit engine(engine_impl* impl):_impl(impl){}
};

class context_impl;
struct context
{
    DLL_SYM context(const context& other);
    DLL_SYM context& operator=(const context& other);
    DLL_SYM static context create();

    DLL_SYM topology create_topology();
    DLL_SYM engine create_engine(uint32_t engine_num, const engine_configuration& configuration);
    DLL_SYM uint32_t engine_count();
private:
    context(context_impl* impl):_impl(impl){}
    context_impl* _impl;
};


}
