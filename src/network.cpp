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
#include "network_builder.h"
#include "api/primitives/input_layout.hpp"
#include "event_impl.h"

namespace cldnn
{
memory_impl* network_impl::get_output_of(const primitive_id& id) const
{
    return _primitives.at(id)->output_memory().get();
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

event_impl* network_impl::execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events)
{
    //TODO implement network execution
    for(auto& evt : events)
    {
        evt->wait();
    }
    auto result = get_engine()->create_user_event();
    result->set();
    return result;
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

memory_impl* network::get_output(primitive_id_ref id, status_t* status) noexcept
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
        return nullptr;
    }
}

engine_impl* network::get_engine_impl(status_t* status) const noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        auto result = _impl->get_engine();
        return result.detach();
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

network_impl* network::build_impl(const engine& engine, const topology& topology, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return engine.get()->build_network(topology);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
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
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>> events(dependencies.size());
        std::transform(dependencies.begin(), dependencies.end(), events.begin(), [](const event& evt) { return refcounted_obj_ptr<event_impl>(evt.get()); });
        return _impl->execute(events);
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}
}
