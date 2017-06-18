/*
// Copyright (c) 2017 Intel Corporation
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

#include "reshape_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "events_waiter.h"
#include "network_impl.h"
#include "reshape/reshape_kernel_selector.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural
{

struct reshape_gpu : public typed_primitive_impl<reshape>
{
    std::unique_ptr<gpu::kernel> _kernel;

    reshape_gpu(reshape_node const& node, const KernelSelector::KernelData& kd)
        : _kernel(std::make_unique<gpu::kernel>(node.get_program().get_engine()->get_context(), kd.kernels[0].kernel_string))
    {
        _use_ks = true;
        _ks_kernel_data = kd;
    }

    reshape_gpu(reshape_node const&){}

    event_impl::ptr execute_impl(std::vector<event_impl::ptr> const& events, reshape_inst& instance)
    {
        if (!_kernel)
        {
            if (events.size() == 1)
                return events[0];

            neural::gpu::events_waiter events_waiter(instance.get_network().get_engine()->get_context());
            return events_waiter.run(events);
        }

        gpu::kernel::kernel_arguments_desc args;
        args.inputs = { &instance.input_memory() };
        args.output = &instance.output_memory();

        return _kernel->run_ks(_ks_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(reshape_node const& node) 
    { 
        if (node.is_in_place())
        {
            return new reshape_gpu(node);
        }

        auto reorder_params = GetDefaultParams<KernelSelector::ReorderBaseParams>(node);
        auto reorder_optional_params = GetDefaultOptionalParams<KernelSelector::ReorderOptionalParams>(node.get_program());

        auto& kernel_selector = KernelSelector::ReshapeKernelSelctor::instance();
        auto best_kernels = kernel_selector.GetBestKernels(reorder_params, reorder_optional_params);
        if (best_kernels.empty())
        {
            throw std::runtime_error("Unsupported - didn't find a proper kernel for this arguments");
        }

        auto reorder = new reshape_gpu(node, best_kernels[0]);

        return reorder;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<reshape>::add({
                { cldnn::engine_types::ocl, reshape_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
