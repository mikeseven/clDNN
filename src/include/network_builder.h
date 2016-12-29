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
#include <set>

namespace cldnn
{

class network_builder
{
public:
    network_builder(refcounted_obj_ptr<engine_impl> eng, const build_options& options)
        : _engine(eng)
        , _options(options)
    {
    }

    network_impl* build_network(refcounted_obj_ptr<topology_impl> tpl)
    {
        assert(tpl);
        _topology_map = tpl->get_primitives();
        
        optimize_topology();
        auto network_topology = refcounted_obj_ptr<topology_impl>(new topology_impl(_topology_map), false);

        auto outputs_option = _options.get<build_option_type::outputs>();
        assert(outputs_option && !outputs_option->outputs.empty());
        
        return new network_impl(get_engine(), network_topology, outputs_option->outputs);
    }

    const refcounted_obj_ptr<engine_impl>& get_engine() const { return _engine; }

private:
    const refcounted_obj_ptr<engine_impl> _engine;
    build_options _options;

    topology_map _topology_map;

    void optimize_topology()
    {
        // in debug mode select all primitives as output
        if(_options.get<build_option::debug>())
        {
            std::vector<primitive_id> outputs;
            for(auto& p : _topology_map)
            {
                outputs.push_back(p.second->id());
            }
            _options.set_option(build_option::outputs(outputs));
            return;
        }

        // TODO some optimizations aka weights reordering, fusing, etc.

        auto outputs_option = _options.get<build_option_type::outputs>();
        if( outputs_option == nullptr || outputs_option->outputs.empty() )
        {
            std::vector<primitive_id> outputs;
            // by default, outputs are primitives which are not inputs for others
            std::set<primitive_id> unreferenced_ids;
            for (auto& pair : _topology_map)
            {
                unreferenced_ids.insert(pair.second->id());
            }
            for (auto& pair : _topology_map)
            {
                for (auto& in : pair.second->dependecies())
                {
                    unreferenced_ids.erase(in);
                }
            }
            std::copy(std::begin(unreferenced_ids), std::end(unreferenced_ids), std::back_inserter(outputs));
            _options.set_option(build_option::outputs(outputs));
        }
    }
};
}
