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
#include "api/topology.hpp"
#include "api/cldnn.hpp"
#include "refcounted_obj.h"
#include "topology_impl.h"
#include "engine_impl.h"
#include "network_impl.h"
#include <algorithm>
#include "network_builder.h"

namespace cldnn
{

context engine::get_context()
{
    return _impl->get_context();
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

buffer* engine::allocate_buffer(layout layout, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return _impl->allocate_buffer(layout);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

network_impl* engine::build_network_impl(topology topology, status_t* status) noexcept
{
    if (topology.get_context() != get_context())
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }

    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        network_builder builder(*this, _impl->configuration());
        return builder.build_network(topology);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}
}
