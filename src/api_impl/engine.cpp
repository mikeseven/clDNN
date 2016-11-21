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
#include "refcounted_obj.h"
#include "topology_impl.h"

namespace cldnn
{
class network_impl : public refcounted_obj<network_impl>
{
public:

    network_impl(const engine& engine, const topology& topology):_topology(topology), _engine(engine)
    {
        build_network();
    }
private:
    topology _topology;
    engine _engine;
    void build_network()
    {
        auto primitives_desc = _topology._impl->get_primitives();
        for(auto p: primitives_desc)
        {
            
        }
    }
};

class engine_impl : public refcounted_obj<engine_impl>
{
    
};

memory engine::allocate_memory(layout layout)
{
    
}

network engine::build_network(topology topology)
{
    if (topology.get_context()._impl != get_context()._impl)
        throw std::runtime_error("topology made in wrong context");
    return new network_impl(*this, topology);
}

context engine::get_context()
{
    
}

engine::engine(const engine& other):_impl(other._impl)
{
    _impl->add_ref();
}

engine& engine::operator=(const engine& other)
{
    if (_impl == other._impl) return *this;
    _impl->release();
    _impl = other._impl;
    _impl->add_ref();
    return *this;
}


}
