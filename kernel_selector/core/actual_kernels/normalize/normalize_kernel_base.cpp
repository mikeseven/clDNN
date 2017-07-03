﻿/*
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

#include "normalize_kernel_base.h"

namespace KernelSelector 
{
    JitConstants NormalizeKernelBase::GetJitConstants(const NormalizeParams& params) const
    {
        return MakeNormalizeJitConstants(params);
    }

    NormalizeKernelBase::DispatchData NormalizeKernelBase::SetDefault(const NormalizeParams& params) const
    {
        const auto& output = params.output;

        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        if (params.normParams.normMode == NormalizeMode::WITHIN_SPATIAL)
        {
            kd.gws0 = output.X().v;
            kd.gws1 = output.Y().v;
            kd.gws2 = output.Batch().v;
        }
        else
        {
            kd.gws0 = output.Batch().v;
            kd.gws1 = 1;
            kd.gws2 = 1;
        }

        kd.lws0 = kd.gws0;
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }

    KernelsData NormalizeKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams&, float estimated_time) const
    {
        assert(params.GetType() == KernelType::NORMALIZE);

        const NormalizeParams& orgParams = static_cast<const NormalizeParams&>(params);

        if (!CheckActivationSupport(orgParams.activationFunc))
        {
            return{};
        }

        DispatchData runInfo;

        try
        {
            runInfo = SetDefault(orgParams);
        }
        catch (const std::runtime_error&)
        {
            return{};
        }

        KernelData kd = KernelData::Default<NormalizeParams>(params, 1);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);
        kernel.argsDesc.data.push_back({ ArgumentDescriptor::Types::SCALE_TABLE, 0 });

        kd.estimatedTime = estimated_time;

        return{ kd };
    }
}