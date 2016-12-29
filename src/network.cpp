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
#include "api/primitives/input_layout.hpp"
#include "network_impl.h"
#include "engine_impl.h"
#include "event_impl.h"
#include "network_builder.h"
#include "primitive_type.h"
#include "input_layout_arg.h"
#include <algorithm>

namespace cldnn
{
network_impl::network_impl(refcounted_obj_ptr<engine_impl> engine, refcounted_obj_ptr<topology_impl> topology, const std::vector<primitive_id>& outputs)
    : _engine(engine)
    , _topology(topology)
    , _output_ids(outputs)
{
    for (auto& output : _output_ids)
    {
        auto p = get_primitive(output);
        assert(p);
    }

    for (auto& p : _primitives)
    {
        if (p.second->type() == input_layout::type_id())
        {
            _input_names.insert({ p.second->id(), false });
        }
    }
}

void network_impl::reset_execution(bool wait)
{
    if (wait)
    {
        for (auto& p : _events)
        {
            p.second->wait();
        }
    }
    _outputs.clear();
    _events.clear();
}

void network_impl::set_input_data(const primitive_id& id, const memory& data)
{
    auto& primitive = _primitives.at(id);
    if (primitive->type() != input_layout::type_id()) throw std::invalid_argument("primitive " + id + " is not an input");

    auto input_c = std::static_pointer_cast<const input_layout_arg>(primitive);
    auto input = std::const_pointer_cast<input_layout_arg>(input_c);

    reset_execution(true);
    input->set_data(data);
    _input_names[input->id()] = true;
}

array_ref<network::network_output_ref> network_impl::execute(const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    //Wait for pervious execution completion
    reset_execution(true);

    for(auto& output_id : _output_ids)
    {
        auto primitive = get_primitive(output_id);
        auto output_event = execute_primitive(primitive, events);
        auto output_memory = primitive->output_memory();
        _outputs.push_back({ output_id, output_event.get(), output_memory.get() });
    }
    return _outputs;
}

std::shared_ptr<const primitive_arg> network_impl::get_primitive(const primitive_id& id)
{
    auto it = _primitives.find(id);
    if (it != _primitives.end())
    {
        return it->second;
    }

    auto& desc = _topology->at(id);
    auto primitive = desc->type()->create_arg(*this, desc);
    return _primitives.insert({ id, primitive }).first->second;
}

std::vector<std::shared_ptr<const primitive_arg>> network_impl::get_primitives(const std::vector<primitive_id>& ids)
{
    std::vector<std::shared_ptr<const primitive_arg>> result(ids.size());
    std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) { return get_primitive(id); });
    return result;
}

refcounted_obj_ptr<event_impl> network_impl::execute_primitive(const std::shared_ptr<const primitive_arg>& primitive, const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    auto it = _events.find(primitive->id());
    if(it != _events.end())
    {
        return it->second;
    }
    return _events.insert({ primitive->id(), primitive->execute(events) }).first->second;
}

// cldnn::network
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

topology_impl* network::get_topology_impl(status_t* status) const noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        auto topology = _impl->get_topology();
        return topology.detach();
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

network_impl* network::build_impl(const engine& engine, const topology& topology, array_ref<build_option_ref> options, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        build_options opts(options);
        return engine.get()->build_network(topology, options);
    }
    catch(const std::exception& e)
    {
        e.what();
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
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

array_ref<network::network_output_ref> network::execute_impl(array_ref<event> dependencies, status_t* status) noexcept
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
        return array_ref<network::network_output_ref>();
    }
}
}
