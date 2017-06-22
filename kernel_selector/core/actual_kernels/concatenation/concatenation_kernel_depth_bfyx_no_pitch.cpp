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

#include "concatenation_kernel_depth_bfyx_no_pitch.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{

    ParamsKey ConcatenationKernel_depth_bfyx_no_pitch::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetInputLayout(DataLayout::bf);
        k.SetOutputLayout(DataLayout::bfyx);
        k.SetOutputLayout(DataLayout::bf);
        k.SetOffsetSupport();
        k.SetBatchingSupport();
        k.SetConcatAxis(ConcatAxis::FEATURE);
        return k;
    }

    IGKConcatenationKernelBase::DispatchData ConcatenationKernel_depth_bfyx_no_pitch::SetDefault(const ConcatenationParams& params) const
    {
        DispatchData runInfo = IGKConcatenationKernelBase::SetDefault(params);
        const auto& input = params.inputs[0];
        const auto batch = input.Batch().v;
        runInfo.gws0 = batch;
        runInfo.gws1 = cldnn::align_to(std::max((size_t)1, input.Length() / batch / 8), 16);
        runInfo.gws2 = 1;

        runInfo.lws0 = 1;
        runInfo.lws1 = 16;
        runInfo.lws2 = 1;
        
        runInfo.effiency = FORCE_PRIORITY_9;

        return runInfo;
    }

    KernelsData ConcatenationKernel_depth_bfyx_no_pitch::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::CONCATENATION);

        const ConcatenationParams& orgParams = static_cast<const ConcatenationParams&>(params);

        const bool bSupportedActivation = orgParams.activationFunc == ActivationFunction::NONE;
        
        if (!bSupportedActivation)
        {
            return{};
        }

        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<ConcatenationParams>(params);

        auto cldnnJit = GetJitConstants(orgParams);
        auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entryPoint);

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}