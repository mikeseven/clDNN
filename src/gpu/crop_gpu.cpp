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

#include "crop_inst.h"
#include "kernel.h"
#include "events_waiter.h"
#include "implementation_map.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "kernel_selector_helper.h"

using namespace cldnn;
using namespace KernelSelector;

namespace neural
{

struct crop_gpu : typed_primitive_impl<crop>
{
    const crop_node& outer;
    gpu::kernel _kernel;

    crop_gpu(const crop_node& arg, const KernelData& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _use_ks = true;
        _ks_kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, crop_inst& instance) override
    {
        if (outer.can_be_optimized())
        {
            if (events.size() == 1)
                return events[0];

            return neural::gpu::events_waiter(outer.get_program().get_engine()->get_context()).run(events);
        }

        gpu::kernel::kernel_arguments_desc args;
        args.inputs = { &instance.input_memory() };
        args.output = &instance.output_memory();

        return _kernel.run_ks(_ks_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const crop_node& arg) 
    { 
        auto ew_params = GetDefaultParams<EltwiseParams>(arg, 1);
        auto ew_optional_params = GetDefaultOptionalParams<EltwiseOptionalParams>(arg.get_program());

        ew_params.eltwiseParams.operations.push_back({{ EltwiseParams::InputType::Buffer(0) }, EltwiseMode::ASSIGN });

        const auto& input_layout = arg.input().get_output_layout();
        ew_params.inputs[0] = ConvertDataTensor(input_layout, 1, arg.get_primitive()->offsets);

        auto& kernel_selector = EltwiseKernelSelctor::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Unsupported - didn't find a proper kernel for this arguments");
        }

        auto crop = new crop_gpu(arg, best_kernels[0]);

        return crop;
    };
};

namespace {
    struct attach {
        attach() {
            auto val_fw = crop_gpu::create;

            implementation_map<crop>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<crop>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<crop>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<crop>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        }
        ~attach() {}
    };

    attach attach_impl;

}
} // namespace neural
