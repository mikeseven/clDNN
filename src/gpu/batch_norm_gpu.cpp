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

#include "batch_norm_inst.h"
#include "kernel.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "kernel_selector_helper.h"

using namespace cldnn;
using namespace KernelSelector;

namespace neural
{

struct batch_norm_gpu : typed_primitive_impl<batch_norm>
{
    const batch_norm_node& outer;
    gpu::kernel _kernel;

    batch_norm_gpu(const batch_norm_node& arg, const KernelData& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _use_ks = true;
        _ks_kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, batch_norm_inst& instance) override
    {
        gpu::kernel::kernel_arguments_desc args;
        args.inputs = { &instance.input_memory(), &instance.mean_memory(), &instance.variance_memory() };
        args.output = &instance.output_memory();

        return _kernel.run_ks(_ks_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const batch_norm_node &arg) 
    { 
        if (arg.get_primitive()->use_global_stats == false)
        {
            throw std::runtime_error("no_global_stats is not supported - it's for training only.");
        }

        auto ew_params = GetDefaultParams<EltwiseParams>(arg);
        auto ew_optional_params = GetDefaultOptionalParams<EltwiseOptionalParams>(arg.get_program());

        ew_params.inputs.push_back(ConvertDataTensor(arg.mean().get_output_layout()));
        ew_params.inputs.push_back(ConvertDataTensor(arg.variance().get_output_layout()));

        ew_params.eltwiseParams.operations.push_back({
            { EltwiseParams::InputType::Buffer(0), EltwiseParams::InputType::Buffer(1) },
            EltwiseMode::SUB });

        ew_params.eltwiseParams.operations.push_back({
            { EltwiseParams::InputType::Buffer(2), EltwiseParams::InputType::Scalar(arg.get_primitive()->epsilon) },
            EltwiseMode::ADD });

        ew_params.eltwiseParams.operations.push_back({
            { EltwiseParams::InputType::Intermediate(1) },
            EltwiseMode::SQRT });

        ew_params.eltwiseParams.operations.push_back({
            { EltwiseParams::InputType::Intermediate(0), EltwiseParams::InputType::Intermediate(2) },
            EltwiseMode::DIV });

        ew_params.eltwiseParams.layoutBased = true;

        auto& kernel_selector = EltwiseKernelSelctor::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Unsupported - didn't find a proper kernel for this arguments");
        }

        auto norm = new batch_norm_gpu(arg, best_kernels[0]);

        return norm;
    };
};

namespace {
    struct attach {
        attach() {
            auto val_fw = batch_norm_gpu::create;

            implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} // namespace neural
