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

#include "ks_reorder_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"
#include "kernel_selector_common.h"
#include "network_impl.h"
#include "engine_impl.h"
#include <algorithm>

using namespace cldnn;
using WeightsReorderParams = KernelSelector::WeightsReorderParams;

namespace neural
{

struct ks_reorder_gpu : typed_primitive_impl<ks_reorder>
{
    const ks_reorder_node& outer;
    gpu::engine_info_internal _engine_info;
    gpu::kernel* _kernel_ptr;

    ks_reorder_gpu(const ks_reorder_node& arg)
    : outer(arg)
    , _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info())
    , _kernel_ptr(nullptr)
    {
        const WeightsReorderParams* reorder_params = outer.get_primitive()->reorder_params;
        if (reorder_params->engine == WeightsReorderParams::Engine::GPU)
        {
            engine_impl* eimpl = arg.get_program().get_engine().get();
            auto context = eimpl->get_context();
            const auto& kernel_string = reorder_params->cl_kernel.kernel_string;
            _kernel_ptr = new gpu::kernel(context, kernel_string);
        }
    }

    virtual ~ks_reorder_gpu()
    {
        if (_kernel_ptr)
        {
            delete _kernel_ptr;
            _kernel_ptr = nullptr;
        }
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, ks_reorder_inst& instance) override
    {
        auto& input_mem = instance.input_memory();
        auto& output_mem = instance.output_memory();

        std::vector<event_impl::ptr> tmp_events(events);

        const WeightsReorderParams* reorder_params = outer.get_primitive()->reorder_params;
        if (reorder_params->engine == WeightsReorderParams::Engine::CPU)
        {
            for (auto& a : events) {
                a->wait();
            }

            auto  old_pointer = input_mem.pointer<uint8_t>();;
            auto  new_pointer = output_mem.pointer<uint8_t>();;

            reorder_params->cpu_kernel->Execute(old_pointer.data(), old_pointer.size(), new_pointer.data(), new_pointer.size());

            event_impl* ev = instance.get_network().get_engine()->create_user_event();
            ev->set();
            tmp_events.emplace_back(ev);
        }
        else if (reorder_params->engine == WeightsReorderParams::Engine::GPU)
        {
            auto event = _kernel_ptr->run_ks(
                reorder_params->cl_kernel,
                events,
                { &input_mem },
                &output_mem);
            tmp_events.emplace_back(event);
        }

        return tmp_events.at(0);
    }

    static primitive_impl* create(const ks_reorder_node& arg)
    {
        return new ks_reorder_gpu(arg);
    }
};

namespace {
    struct attach {
        attach() {
            auto val_fw = ks_reorder_gpu::create;
            implementation_map<ks_reorder>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw);
            implementation_map<ks_reorder>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<ks_reorder>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<ks_reorder>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            //implementation_map<ks_reorder>::add({ { cldnn::engine_types::ocl, val_fw } });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}