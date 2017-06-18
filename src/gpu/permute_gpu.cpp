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

#include "permute_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "permute/permute_kernel_selector.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural
{

struct permute_gpu : typed_primitive_impl<permute>
{
    const permute_node& outer;
    gpu::kernel _kernel;

    permute_gpu(const permute_node& arg, const KernelSelector::KernelData& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernel_string)
    {
        _use_ks = true;
        _ks_kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, permute_inst& instance) override
    {
        gpu::kernel::kernel_arguments_desc args;
        args.inputs = { &instance.input_memory() };
        args.output = &instance.output_memory();

        return _kernel.run_ks(_ks_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const permute_node& arg)
    {
        auto reorder_params = GetDefaultParams<KernelSelector::PermuteParams>(arg);
        auto reorder_optional_params = GetDefaultOptionalParams<KernelSelector::ReorderOptionalParams>(arg.get_program());
        uint16_t max_input_index = (uint16_t)(reorder_params.inputs[0].dims.size() - 1);
        const auto& permute_order = arg.get_primitive()->permute_order;
        for (size_t i = 0; i < permute_order.size(); i++)
        {
            auto order = permute_order[permute_order.size() - 1 - i];
            reorder_params.permuteParams.order.push_back(max_input_index - order);
        }
        auto& kernel_selector = KernelSelector::PermuteKernelSelctor::instance();
        auto best_kernels = kernel_selector.GetBestKernels(reorder_params, reorder_optional_params);
        if (best_kernels.empty())
        {
            throw std::runtime_error("Unsupported - didn't find a proper kernel for this arguments");
        }

        auto reorder = new permute_gpu(arg, best_kernels[0]);

        return reorder;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<permute>::add({
                { cldnn::engine_types::ocl, permute_gpu::create },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}