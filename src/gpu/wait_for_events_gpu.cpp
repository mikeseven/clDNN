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

#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "data_inst.h"
#include "prior_box_inst.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "events_waiter.h"

namespace cldnn
{
class wait_for_events_gpu : public primitive_impl
{
    network_impl& _network;

public:
    wait_for_events_gpu(primitive_inst* primitive) : _network(primitive->get_network()) {}

    refcounted_obj_ptr<event_impl> execute(const std::vector<refcounted_obj_ptr<event_impl>>& events) override
    {
        neural::gpu::events_waiter events_waiter(_network.get_engine()->get_context());
        return events_waiter.run(events);
    }

    static primitive_impl* create_data(data_inst& data)
    {
        return new wait_for_events_gpu(&data);
    }

    static primitive_impl* create_input_layout(input_layout_inst& input)
    {
        return new wait_for_events_gpu(&input);
    }

	static primitive_impl* create_prior_box(prior_box_inst& prior_box)
	{
		return new wait_for_events_gpu(&prior_box);
	}
};

namespace {
    struct attach {
        attach() {
            implementation_map<data_inst>::add({
                { engine_types::ocl, wait_for_events_gpu::create_data }
            });

            implementation_map<input_layout_inst>::add({
                { engine_types::ocl, wait_for_events_gpu::create_input_layout }
            });

			implementation_map<prior_box_inst>::add({
				{ engine_types::ocl, wait_for_events_gpu::create_prior_box }
			});
        }
        ~attach() {}
    };
    attach attach_impl;
}

}
