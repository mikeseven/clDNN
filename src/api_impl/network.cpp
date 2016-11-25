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
#include "network_impl.h"
#include "engine_impl.h"
#include <algorithm>

namespace cldnn
{

std::shared_ptr<const primitive_arg> network_impl::get_primitive(primitive_id id)
{
    auto it = _primitives.find(id);
    if (it != _primitives.end())
        return it->second;

    auto& desc = _topology.implementation()->get_primitives().at(id);
    return _primitives.insert({ id, desc->get_dto()->type->create_arg(this, desc) }).first->second;
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

const memory& network::get_output(primitive_id_ref id)
{
}

engine network::get_engine()
{
    return _impl->get_engine();
}

status_t network::set_input_data_impl(primitive_id_ref id, memory mem)
{
}

array_ref<primitive_id_ref> network::get_primitive_keys_impl(status_t* status)
{
}

event_impl* network::execute_impl(array_ref<event> dependencies, status_t* status)
{
}
}
