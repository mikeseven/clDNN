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

#include "igk_normalize_kernel_base.h"

namespace KernelSelector 
{
    jit_constants IGKNormalizeKernelBase::GetJitConstants(const NormalizeParams& params) const
    {
        gpu::jit_constants mem_consts = GetCommonJitConstants(params);

        auto scale_feature_size = params.normParams.scaleTable.feature().v;

        mem_consts.add_constants({
            gpu::make_jit_constant("SCALE_INDEX",   (scale_feature_size == 1) ? "0" : "f"),
            gpu::make_jit_constant("EPSILON",       params.normParams.epsilon),
        });

        return mem_consts;
    }

    IGKNormalizeKernelBase::DispatchData IGKNormalizeKernelBase::SetDefault(const NormalizeParams& params) const
    {
        const auto& output = params.output;

        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].dtype == Datatype::F16;

        if (params.normParams.normMode == NormalizeMode::WITHIN_SPATIAL)
        {
            kd.gws0 = output.x().v;
            kd.gws1 = output.y().v;
            kd.gws2 = output.batch().v;
        }
        else
        {
            kd.gws0 = output.batch().v;
            kd.gws1 = 1;
            kd.gws2 = 1;
        }

        kd.lws0 = kd.gws0;
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }

    KernelsData IGKNormalizeKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams&, float estimated_time) const
    {
        assert(params.GetType() == KernelType::NORMALIZE);

        const NormalizeParams& orgParams = static_cast<const NormalizeParams&>(params);

        if (!CheckActivationSupport(orgParams.activationFunc))
        {
            return{};
        }

        DispatchData run_info;

        try
        {
            run_info = SetDefault(orgParams);
        }
        catch (const std::runtime_error&)
        {
            return{};
        }

        KernelData kd = KernelData::Default<NormalizeParams>(params, 1);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit.get_definitions(), entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, run_info, kernelName, jit, entry_point);
        kernel.argsDesc.data.push_back({ ArgumentDescpirtor::Types::SCALE_TABLE, 0 });

        kd.estimatedTime = estimated_time;

        return{ kd };
    }
}