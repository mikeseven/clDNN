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

#include "lrn_kernel_within_channel_ref.h"
 
namespace KernelSelector 
{
    ParamsKey LRNKernelWithinChannel::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetInputLayout(DataLayout::yxfb);
        k.SetOutputLayout(DataLayout::bfyx);
        k.SetOutputLayout(DataLayout::yxfb);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        k.SetLRNMode(LRNMode::WITHIN_CHANNEL);
        k.SetLRNKernelDividerMode(KernelDividerMode::DYNAMIC);
        return k;
    }

    CommonDispatchData LRNKernelWithinChannel::defaultWithinChannel(const LRNParams& params) const
    {
        CommonDispatchData runInfo = SetDefault(params);

        runInfo.gws0 = 128 * 128;
        runInfo.gws1 = 1;
        runInfo.gws2 = 1;

        runInfo.lws0 = 128;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        return runInfo;
    }

    KernelsData LRNKernelWithinChannel::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::LRN);

        const LRNParams& orgParams = static_cast<const LRNParams&>(params);

        const bool bSupportedActivation = orgParams.activationFunc == ActivationFunction::NONE;

        if (!bSupportedActivation)
        {
            return{};
        }
        
        DispatchData runInfo;

        try
        {
            runInfo = defaultWithinChannel(orgParams);
        }
        catch (const std::runtime_error&)
        {
            return{};
        }

        KernelData kd = KernelData::Default<LRNParams>(params, 1);

        auto cldnn_jit = GetJitConstants(orgParams, runInfo);
        auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entryPoint);

        kd.estimatedTime = FORCE_PRIORITY_9;

        return{ kd };
    }
}