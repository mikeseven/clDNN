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
#include "primitive_type.h"
#include "primitive_arg.h"
#include "network_impl.h"
#include "engine_impl.h"
#include "topology_impl.h"
#include <map>

namespace cldnn
{

typedef engine_configuration build_settings;

class network_builder
{
public:
    network_builder(const engine& eng, const build_settings& configuration)
        : _engine(eng)
        , _configuration(configuration)
    {
    }

    network_impl* build_network(topology_impl* tpl)
    {
        optimize_topology(tpl);
        _network.clear();
        for (auto& pair:_topology)
        {
            auto p = get_primitive(pair.first);
            assert(p);
        }
        return new network_impl(get_engine(), _network);
    }

    engine get_engine() const { return _engine; }

    std::shared_ptr<const primitive_arg> get_primitive(const primitive_id& id)
    {
        auto it = _network.find(id);
        return (it != _network.end())
            ? it->second
            : new_primitive(id);
    }

private:
    engine _engine;

    build_settings _configuration;
    topology_map _topology;
    std::map<primitive_id, std::shared_ptr<const primitive_arg>> _network;

    void optimize_topology(topology_impl* tpl)
    {
        auto& original_primitives = tpl->get_primitives();
        // TODO instead of copy, do some optimizations aka weights reordering, fusing, etc.
        _topology = original_primitives;
    }

    std::shared_ptr<const primitive_arg> new_primitive(const primitive_id& id)
    {
        auto& desc = _topology.at(id);
        auto primitive = desc->type()->create_arg(*this, desc);
        return _network.insert({ id, primitive }).first->second;
    }
};
}
