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
#include "api/cldnn.hpp"
#include "context_impl.h"
#include "topology_impl.h"
#include "engine_impl.h"

namespace cldnn
{
context::context(const context& other):_impl(other._impl)
{
    _impl->add_ref();
}

context& context::operator=(const context& other)
{
    if (_impl == other._impl) return *this;
    _impl->release();
    _impl = other._impl;
    _impl->add_ref();
    return *this;
}

context::~context()
{
    _impl->release();
}

uint32_t context::engine_count(engine_types)
{
    return 1;
}

context_impl* context::create_context_impl(status_t* status) noexcept
{
    try {
        if (status)
            *status = CLDNN_SUCCESS;
        return new context_impl();
    }
    catch(...)
    {
        if(status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

topology_impl* context::create_topology_impl(status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return new topology_impl(*this);
    }
    catch(...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

engine_impl* context::create_engine_impl(uint32_t engine_num, const engine_configuration* configuration, status_t* status) noexcept
{
    if (configuration && configuration->engine_type != engine_types::ocl)
    {
        if (status)
            *status = CLDNN_UNSUPPORTED;
        return nullptr;
    }

    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return new engine_impl(*this, configuration ? *configuration : engine_configuration());
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}
}
