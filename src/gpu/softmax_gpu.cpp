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

#include "softmax_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "softmax/softmax_kernel_selector.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural
{

struct softmax_gpu : typed_primitive_impl<softmax>
{
    const softmax_node& outer;
    gpu::kernel _kernel;


    softmax_gpu(const softmax_node& arg, const KernelSelector::KernelData& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _use_ks = true;
        _ks_kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, softmax_inst& instance) override
    {
        gpu::kernel::kernel_arguments_desc args;
        args.inputs = { &instance.input_memory() };
        args.output = &instance.output_memory();

        return _kernel.run_ks(_ks_kernel_data.kernels[0], events, args);
    }

    
    static primitive_impl* create(const softmax_node& arg) 
    {
        auto sm_params = GetDefaultParams<KernelSelector::SoftmaxParams>(arg);
        auto sm_optional_params = GetDefaultOptionalParams<KernelSelector::SoftmaxOptionalParams>(arg.get_program());

        auto& input = sm_params.inputs[0];
        auto& output = sm_params.output;
        auto& sm = sm_params.smParams;
        const auto primitive = arg.get_primitive();

        switch (primitive->dimension)
        {
        case softmax::normalize_x:
            sm.dim = KernelSelector::SoftmaxDim::X;
            break;
        case softmax::normalize_y:
            sm.dim = KernelSelector::SoftmaxDim::Y;
            break;
        case softmax::normalize_fyx:
            // W/A for bf/bx issue of cldnn
            input = input.flatten_fyx_2_f();
            output = output.flatten_fyx_2_f();
        case softmax::normalize_f:
            sm.dim = KernelSelector::SoftmaxDim::FEATURE;
            break;
        case softmax::normalize_bfyx:
        case softmax::normalize_yx:
        case softmax::normalize_b:
        default:
            throw std::runtime_error("Wrong API - no such softmax");
        }

        auto& kernel_selector = KernelSelector::SoftmaxKernelSelctor::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(sm_params, sm_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Unsupported - didn't find a proper kernel for this arguments");
        }

        auto softmax_node = new softmax_gpu(arg, best_kernels[0]);

        return softmax_node;
    };
};

namespace {
    struct attach {
        attach() {
            auto val_fw = softmax_gpu::create;
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        }
        ~attach() {}
    };
}
attach attach_impl;
} // namespace neural
