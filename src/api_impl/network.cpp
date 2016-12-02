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
#include "network_impl.h"
#include "engine_impl.h"

namespace cldnn
{
const memory& network_impl::get_output_of(const primitive_id& id) const
{
    return _primitives.at(id)->output_memory();
}

void network_impl::set_input_data(const primitive_id& id, const memory& data)
{
    auto& primitive = _primitives.at(id);
    if (primitive->type() != input_layout::type_id()) throw std::invalid_argument("primitive " + id + " is not an input");
    auto& dest_mem = primitive->output_memory();
    if (dest_mem.get_layout() != data.get_layout()) throw std::invalid_argument("memory layout do not match");

    // TODO find the way to avoid data copy
    pointer<char> src(data);
    pointer<char> dst(primitive->output_memory());
    std::copy(src.begin(), src.end(), dst.begin());
    _completed = false;
    _inputs[id] = true;
}

network::network(const network& other):_impl(other._impl)
{
    _impl->add_ref();
}

network& network::operator=(const network& other)
{
    if (_impl == other._impl) return *this;
    _impl->release();
    _impl = other._impl;
    _impl->add_ref();
    return *this;
}

network::~network()
{
    _impl->release();
}

const memory& network::get_output(primitive_id_ref id, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return _impl->get_output_of(id);;
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return get_engine().allocate_memory({data_types::f32, {format::x, {0}}});
    }
}

engine network::get_engine()
{
    return _impl->get_engine();
}

status_t network::set_input_data_impl(primitive_id_ref id, memory mem) noexcept
{
    try
    {
        _impl->set_input_data(id, mem);
        return CLDNN_SUCCESS;
    }
    catch (...)
    {
        return CLDNN_ERROR;
    }
}

array_ref<primitive_id_ref> network::get_primitive_keys_impl(status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return _impl->get_primitive_keys();
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return array_ref<primitive_id_ref>();
    }
}

event_impl* network::execute_impl(array_ref<event> dependencies, status_t* status) noexcept
{
}
}
