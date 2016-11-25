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
#include <map>
#include "api/cldnn.hpp"
#include "refcounted_obj.h"
#include "topology_impl.h"
#include "api/reorder.hpp"
#include "api/convolution.hpp"

namespace cldnn
{
class network_impl : public refcounted_obj<network_impl>
{
public:

    network_impl(const engine& engine, const topology& topology):_topology(topology), _engine(engine)
    {
        build_network();
    }

    std::shared_ptr<const primitive_arg> get_primitive(primitive_id id);

    engine get_engine() const { return _engine; }

private:
    topology _topology;
    std::map<primitive_id, std::shared_ptr<const primitive_arg>> _primitives;
    engine _engine;
    void build_network()
    {
        for(auto p: _topology.implementation()->get_primitives())
        {
            
        }
    }
};


}
