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
#include "api/network.hpp"
#include "engine_impl.h"
#include "topology_impl.h"
#include "refcounted_obj.h"
#include "primitive_arg.h"
#include <map>
#include <vector>

namespace cldnn
{
class network_impl : public refcounted_obj<network_impl>
{
public:
    typedef std::map<primitive_id, std::shared_ptr<const primitive>> topology_map;

    network_impl(refcounted_obj_ptr<engine_impl> engine, refcounted_obj_ptr<topology_impl> topology, const std::vector<primitive_id>& outputs);

    const refcounted_obj_ptr<engine_impl>& get_engine() const { return _engine; }
    const refcounted_obj_ptr<topology_impl>& get_topology() const { return _topology; }

    void reset_execution(bool wait = true);
    void set_input_data(const primitive_id& id, const memory& data);
    array_ref<network::network_output_ref> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events);

    // Implementation specific calls
    std::shared_ptr<const primitive_arg> get_primitive(const primitive_id& id);
    std::vector<std::shared_ptr<const primitive_arg>> get_primitives(const std::vector<primitive_id>& ids);
    refcounted_obj_ptr<event_impl> execute_primitive(const std::shared_ptr<const primitive_arg>& primitive, const std::vector<refcounted_obj_ptr<event_impl>>& events);

private:
    const refcounted_obj_ptr<engine_impl> _engine;
    const refcounted_obj_ptr<topology_impl> _topology;
    std::vector<primitive_id> _output_ids;
    std::map<primitive_id, std::shared_ptr<const primitive_arg>> _primitives;
    std::map<primitive_id, bool> _input_names;
    std::map<primitive_id, refcounted_obj_ptr<event_impl>> _events;
    std::vector<network::network_output_ref> _outputs;
};
}
