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

#include "primitive_arg.h"
#include "input_layout_arg.h"
#include "data_arg.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "events_waiter.h"

namespace cldnn
{
class wait_for_events_gpu : public primitive_impl
{
    network_impl& _network;

public:
    wait_for_events_gpu(primitive_arg* primitive) : _network(primitive->get_network()) {}

    refcounted_obj_ptr<event_impl> execute(const std::vector<refcounted_obj_ptr<event_impl>>& events) override
    {
        neural::gpu::events_waiter events_waiter(_network.get_engine()->get_context());
        return events_waiter.run(events);
    }

    static primitive_impl* create_data(data_arg& data)
    {
        return new wait_for_events_gpu(&data);
    }

    static primitive_impl* create_input_layout(input_layout_arg& input)
    {
        return new wait_for_events_gpu(&input);
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<data_arg>::add({
                { engine_types::ocl, wait_for_events_gpu::create_data }
            });

            implementation_map<input_layout_arg>::add({
                { engine_types::ocl, wait_for_events_gpu::create_input_layout }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}

}
