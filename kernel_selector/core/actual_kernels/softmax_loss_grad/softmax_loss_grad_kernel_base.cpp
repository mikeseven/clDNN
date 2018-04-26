/*
// Copyright (c) 2018 Intel Corporation
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

#include "softmax_loss_grad_kernel_base.h"

namespace KernelSelector 
{
    JitConstants SoftmaxLossGradKernelBase::GetJitConstants(const SoftmaxLossGradParams& params) const
    {
        return MakeSoftmaxLossGradJitConstants(params);
    }

    CommonDispatchData SoftmaxLossGradKernelBase::SetDefault(const SoftmaxLossGradParams& params, const OptionalParams&) const
    {
        CommonDispatchData runInfo;

        runInfo.gws0 = 1;
        runInfo.gws1 = 1;
        runInfo.gws2 = 1;

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        runInfo.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        return runInfo;
    }

    bool SoftmaxLossGradKernelBase::Validate(const Params& p, const OptionalParams& o) const
    {
        if (p.GetType() != KernelType::SOFT_MAX_LOSS_GRAD ||
            o.GetType() != KernelType::SOFT_MAX_LOSS_GRAD)
        {
            return false;
        }

        return true;
    }

    KernelsData SoftmaxLossGradKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const SoftmaxLossGradParams& orgParams = static_cast<const SoftmaxLossGradParams&>(params);
        KernelData kd = KernelData::Default<SoftmaxLossGradParams>(params);

        auto runInfo = SetDefault(orgParams, options);
        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 1 });

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}