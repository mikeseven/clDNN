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
#include "primitive_inst.h"
#include "network_impl.h"
#include "engine_impl.h"
#include "memory_impl.h"

namespace cldnn
{
event_impl::ptr primitive_inst::execute(const std::vector<event_impl::ptr>& events)
{
    if (!_has_valid_input)
        throw std::runtime_error("Cannot execute primitive " + id() + " with invalid/unset input");

    if (_deps.size() == 0)
        return _impl->execute(events, *this);

    std::vector<event_impl::ptr> dependencies;
    dependencies.reserve(_deps.size());

    for(auto& input : _deps)
    {
        dependencies.emplace_back(get_network().execute_primitive(input, events));
    }

    return _impl->execute(dependencies, *this);
}

primitive_inst::primitive_inst(network_impl& network, program_node const& node)
    : _network(network)
    , _node(node)
    , _impl(node.get_selected_impl())
    , _deps(network.get_primitives(desc()->dependecies()))
    , _output(allocate_output())
    , _output_changed(false)
{}

primitive_inst::primitive_inst(network_impl& network, program_node const& node, memory const& buffer)
    : _network(network)
    , _node(node)
    , _impl(node.get_selected_impl())
    , _deps(network.get_primitives(desc()->dependecies()))
    , _output(buffer)
    , _output_changed(false)
{
    auto req_layout = node.get_output_layout();
    if (buffer.get_layout() != req_layout)
        throw std::runtime_error("Provided buffer does not meet primitive's requirements");
}

memory primitive_inst::allocate_output()
{
    auto layout = _node.get_output_layout();
    return api_cast(get_network().get_engine()->allocate_buffer(layout));
}
}
