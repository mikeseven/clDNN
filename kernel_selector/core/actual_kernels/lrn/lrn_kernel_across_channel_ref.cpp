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

#include "lrn_kernel_across_channel_ref.h"
 
namespace KernelSelector 
{
    ParamsKey LRNKernelAcrossChannelRef::GetSupportedKey() const
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
        k.SetLRNMode(LRNMode::ACROSS_CHANNEL);
        k.SetLRNKernelDividerMode(KernelDividerMode::FIXED);
        return k;
    }

    CommonDispatchData LRNKernelAcrossChannelRef::default_across_channel(const LRNParams& params) const
    {
        CommonDispatchData run_info = SetDefault(params);

        if (params.inputs[0].layout == DataLayout::bfyx)
        {
            const auto& out = params.output;
            run_info.gws0 = cldnn::align_to(out.x().v, 32);
            run_info.gws1 = out.y().v;
            run_info.gws2 = out.feature().v * out.batch().v;

            run_info.lws0 = 32;
            run_info.lws1 = 1;
            run_info.lws2 = 1;
        }

        return run_info;
    }

    KernelsData LRNKernelAcrossChannelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::LRN);

        const LRNParams& orgParams = static_cast<const LRNParams&>(params);

        const bool bSupportedActivation = orgParams.activationFunc == ActivationFunction::NONE;

        if (!bSupportedActivation)
        {
            return{};
        }
        
        DispatchData run_info;

        try
        {
            run_info = default_across_channel(orgParams);
        }
        catch (const std::runtime_error&)
        {
            return{};
        }

        KernelData kd = KernelData::Default<LRNParams>(params, 1);

        auto cldnn_jit = GetJitConstants(orgParams, run_info);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, run_info, kernelName, jit, entry_point);

        kd.estimatedTime = FORCE_PRIORITY_9;

        return{ kd };
    }
}