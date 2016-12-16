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
#include "engine_impl.h"
#include "network_impl.h"
#include "network_builder.h"
#include "gpu/ocl_toolkit.h"
#include "event_impl.h"
#include "gpu/memory_gpu.h"

namespace cldnn
{
using gpu_toolkit_config = neural::gpu::configuration;

gpu_toolkit_config convert_configuration(const engine_configuration conf)
{
    gpu_toolkit_config result = gpu_toolkit_config::get();
    result.compiler_options = conf.compiler_options;
    result.enable_profiling = conf.enable_profiling != 0;
    return result;
}

engine_impl::engine_impl(const engine_configuration& conf)
    : _configuration(conf)
    , _context(std::make_shared<gpu_toolkit>(convert_configuration(conf)))
{
}

memory_impl* engine_impl::allocate_buffer(layout layout)
{
    return new neural::gpu::gpu_buffer(this, layout);
}

event_impl* engine_impl::create_user_event()
{
    return new user_event_gpu(cl::UserEvent(get_context()->context()));
}

network_impl* engine_impl::build_network(const topology& topology)
{
    network_builder builder(this, configuration());
    return builder.build_network(topology.get());
}

uint32_t engine::engine_count_impl(engine_types type, status_t* status) noexcept
{
    if (type == engine_types::ocl)
    {
        if (status) *status = CLDNN_SUCCESS;
        return 1;
    }
    else
    {
        if (status) *status = CLDNN_UNSUPPORTED;
        return 0;
    }
}

engine::engine(const engine& other):_impl(other._impl)
{
    _impl->add_ref();
}

engine& engine::operator=(const engine& other)
{
    if (_impl == other._impl) return *this;
    _impl->release();
    _impl = other._impl;
    _impl->add_ref();
    return *this;
}

engine::~engine()
{
    _impl->release();
}

engine_types engine::engine_type() noexcept
{
    return engine_types::ocl;
}

engine_impl* engine::create_engine_impl(engine_types engine_type, uint32_t engine_num, const engine_configuration* configuration, status_t* status) noexcept
{
    if (engine_num > 0 || (engine_type != engine_types::ocl))
    {
        if (status)
            *status = CLDNN_UNSUPPORTED;
        return nullptr;
    }

    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return new engine_impl(configuration ? *configuration : engine_configuration());
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}
}
