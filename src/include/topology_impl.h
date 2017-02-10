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
#include "api_impl.h"
#include "refcounted_obj.h"
#include "api/primitive.hpp"
#include <map>

namespace cldnn
{
struct layout;
struct topology_node
{
public:
    topology_node(std::shared_ptr<const primitive> prim_desc)
        : primitive_desc(prim_desc){}

    void replace(std::shared_ptr<const primitive> new_prim)
    {
        if (primitive_desc == new_prim)
            return;

        primitive_desc = new_prim;
        output_layout = nullptr; //invalidate output_layout if primitive_desc has been changedd
    }

    std::shared_ptr<const primitive> primitive_desc;
    std::unique_ptr<layout> output_layout;
};

typedef std::map<primitive_id, std::shared_ptr<topology_node>> topology_map;

struct topology_impl : public refcounted_obj<topology_impl>
{
public:
    topology_impl(const topology_map& map = topology_map()) : _primitives(map) {}

    void add(std::shared_ptr<const primitive> desc)
    {
        auto id = desc->id();
        auto itr = _primitives.find(id);
        if (itr != _primitives.end())
        {
            if (itr->second->primitive_desc != desc)
                throw std::runtime_error("different primitive with id '" + id + "' exists already");

            //adding the same primitive more than once is not an error
            return;
        }
            
        _primitives.insert({ id, std::make_shared<topology_node>(desc)});
    }

    const std::shared_ptr<topology_node>& at(primitive_id id) const { return _primitives.at(id); }

    const topology_map& get_primitives() const { return _primitives; }

private:
    topology_map _primitives;
};
}

API_CAST(::cldnn_topology, cldnn::topology_impl)
