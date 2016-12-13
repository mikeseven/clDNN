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
    network_builder(refcounted_obj_ptr<engine_impl> eng, const build_settings& configuration)
        : _engine(eng)
        , _configuration(configuration)
    {
    }

    network_impl* build_network(topology_impl* tpl)
    {
        assert(tpl);
        return new network_impl(get_engine(), optimize_topology(tpl));
    }

    const refcounted_obj_ptr<engine_impl>& get_engine() const { return _engine; }

private:
    const refcounted_obj_ptr<engine_impl> _engine;

    build_settings _configuration;
    std::map<primitive_id, std::shared_ptr<const primitive_arg>> _network;

    topology_map optimize_topology(topology_impl* tpl)
    {
        auto& original_primitives = tpl->get_primitives();
        // TODO do some optimizations aka weights reordering, fusing, etc.
        return original_primitives;
    }
};
}
