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
#include "api/engine.hpp"
#include "api/network.hpp"
#include "engine_impl.h"
#include "refcounted_obj.h"
#include "primitive_arg.h"
#include "primitive_type.h"
#include "api/primitives/input_layout.hpp"
#include <map>
#include <algorithm>

namespace cldnn
{
class network_impl : public refcounted_obj<network_impl>
{
public:
    typedef std::map<primitive_id, std::shared_ptr<const primitive>> topology_map;

    network_impl(refcounted_obj_ptr<engine_impl> engine, const topology_map& topology)
        : _completed(false)
        , _engine(engine)
        , _topology(topology)
    {
        for (auto& pair : _topology)
        {
            auto p = get_primitive(pair.first);
            assert(p);
        }

        for(auto& p : _primitives)
        {
            if(p.second->type() == input_layout::type_id())
            {
                _inputs.insert({ p.second->id(), false });
            }
        }
    }

    const refcounted_obj_ptr<engine_impl>& get_engine() const noexcept { return _engine; }

    memory_impl* get_output_of(const primitive_id& id) const;
    array_ref<primitive_id_ref> get_primitive_keys() const { return _primitive_names; }
    void set_input_data(const primitive_id& id, const memory& data);

    std::shared_ptr<const primitive_arg> get_primitive(const primitive_id& id)
    {
        auto it = _primitives.find(id);
        return (it != _primitives.end())
            ? it->second
            : new_primitive(id);
    }

    std::vector<std::shared_ptr<const primitive_arg>> get_primitives(const std::vector<primitive_id>& ids)
    {
        std::vector<std::shared_ptr<const primitive_arg>> result(ids.size());
        std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) { return get_primitive(id); });
        return result;
    }

    event_impl* execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events);
private:
    bool _completed;
    const refcounted_obj_ptr<engine_impl> _engine;
    const topology_map _topology;
    std::map<primitive_id, std::shared_ptr<const primitive_arg>> _primitives;
    std::vector<primitive_id_ref> _primitive_names;
    std::map<primitive_id, bool> _inputs;

    static std::vector<primitive_id_ref> get_primitive_names(const std::map<primitive_id, std::shared_ptr<const primitive_arg>>& primitives)
    {
        std::vector<primitive_id_ref> result;
        for(auto& pair: primitives)
        {
            // it should be reference to the constant primitive_id store.
            result.push_back(pair.second->id());
        }
        return result;
    }

    std::shared_ptr<const primitive_arg> new_primitive(const primitive_id& id)
    {
        auto& desc = _topology.at(id);
        auto primitive = desc->type()->create_arg(*this, desc);
        return _primitives.insert({ id, primitive }).first->second;
    }
};
}
