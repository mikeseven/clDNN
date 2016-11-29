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
#include "topology.hpp"
#include "engine.hpp"

namespace cldnn
{
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
