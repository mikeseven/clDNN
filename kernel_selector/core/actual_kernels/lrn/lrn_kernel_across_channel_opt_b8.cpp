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

#include "lrn_kernel_across_channel_opt_b8.h"
 
namespace KernelSelector 
{
    ParamsKey LRNKernelAcrossChannel_b8::GetSupportedKey() const
    {
        ParamsKey k;
        //k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        //k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::yxfb);
        k.SetOutputLayout(DataLayout::yxfb);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        k.SetLRNMode(LRNMode::ACROSS_CHANNEL);
        k.SetLRNKernelDividerMode(KernelDividerMode::FIXED);
        k.SetSubGroupSupport();
        return k;
    }

    CommonDispatchData LRNKernelAcrossChannel_b8::default_across_channel_b8(const LRNParams& params) const
    {
        CommonDispatchData run_info = SetDefault(params);

        run_info.gws0 /= 8;
        run_info.lws0 = 8; // gws0 is dividable by 64, so after correction it will be dividable by 8.

        return run_info;
    }

    KernelsData LRNKernelAcrossChannel_b8::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::LRN);

        const LRNParams& orgParams = static_cast<const LRNParams&>(params);
        const auto& out = orgParams.output;

        const bool bSupportedActivation = orgParams.activationFunc == ActivationFunction::NONE;
        const bool bSupportedPitch =
            orgParams.inputs[0].batch().pitch == 1 &&
            out.batch().pitch == 1;
        const bool bSupportedBatch = 
            (out.batch().v % 8) == 0 &&
            ((out.batch().v * out.feature().v) % 64) == 0;

        if (!bSupportedActivation || !bSupportedPitch || !bSupportedBatch)
        {
            return{};
        }
        
        DispatchData run_info;

        try
        {
            run_info = default_across_channel_b8(orgParams);
        }
        catch (const std::runtime_error&)
        {
            return{};
        }

        KernelData kd = KernelData::Default<LRNParams>(params, 1);

        auto cldnn_jit = GetJitConstants(orgParams, run_info);
        
        cldnn_jit.add_constant(gpu::make_jit_constant("SUB_GROUP_SIZE", 8));
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit.get_definitions(), entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, run_info, kernelName, jit, entry_point);

        kd.estimatedTime = FORCE_PRIORITY_9;

        return{ kd };
    }
}