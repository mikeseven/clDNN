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
#include "refcounted_obj.h"
#include "primitive_arg.h"
#include "implementation_map.h"
#include "gpu/engine_info.h"

#include <memory>

namespace neural { namespace gpu { class gpu_toolkit; } }
namespace cldnn
{
class build_options;
using gpu_toolkit = neural::gpu::gpu_toolkit;
struct topology_impl;
struct memory_impl;
struct engine_impl : public refcounted_obj<engine_impl>
{
public:
    engine_impl(const engine_configuration& conf);

    engine_types type() const { return engine_types::ocl; }

    memory_impl* allocate_buffer(layout layout);
    event_impl* create_user_event();
    network_impl* build_network(topology_impl* topology, const build_options& options);
    const engine_configuration& configuration() const { return _configuration; }

    std::shared_ptr<gpu_toolkit> get_context() const { return _context; }

    // TODO the following function should be refactored. Psbl use one map instance for all primitives
    template<class primitive_kind>
    std::unique_ptr<primitive_impl> create_primitive_impl(primitive_kind& arg)
    {
        auto factory = implementation_map<primitive_kind>::get(type(), arg);
        return std::move(std::unique_ptr<primitive_impl>(factory(arg)));
    }

    neural::gpu::engine_info_internal get_engine_info() const
    {
        return _context->get_engine_info();
    }

private:
    engine_configuration _configuration;
    std::shared_ptr<gpu_toolkit> _context;
};
}

API_CAST(::cldnn_engine, cldnn::engine_impl)
