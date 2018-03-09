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

#include "lookup_table_kernel_axis.h"

namespace KernelSelector
{
    ParamsKey LookUpTableKernelAxis::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableLookUpTableAxis(LookUpTableAxis::BATCH);
        k.EnableLookUpTableAxis(LookUpTableAxis::X);
        k.EnableLookUpTableAxis(LookUpTableAxis::Y);
        k.EnableLookUpTableAxis(LookUpTableAxis::FEATURE);
        k.EnableBatching();
        return k;
    }

    KernelsData LookUpTableKernelAxis::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const LookUpTableParams& orgParams = static_cast<const LookUpTableParams&>(params);

        DispatchData runInfo;
        runInfo.fp16UnitUsed = orgParams.inputs[0].GetDType() == Datatype::F16;

        if (orgParams.lookUpTableParams.lookUpTableAxis == LookUpTableAxis::BATCH) {
            runInfo.gws0 = orgParams.inputs[0].X().v;
            runInfo.gws1 = orgParams.inputs[0].Y().v;
            runInfo.gws2 = orgParams.inputs[0].Feature().v;
        }
        else if (orgParams.lookUpTableParams.lookUpTableAxis == LookUpTableAxis::FEATURE) {
            runInfo.gws0 = orgParams.inputs[0].X().v;
            runInfo.gws1 = orgParams.inputs[0].Y().v;
            runInfo.gws2 = orgParams.inputs[0].Batch().v;
        }
        else if (orgParams.lookUpTableParams.lookUpTableAxis == LookUpTableAxis::Y) {
            runInfo.gws0 = orgParams.inputs[0].X().v;
            runInfo.gws1 = orgParams.inputs[0].Feature().v;
            runInfo.gws2 = orgParams.inputs[0].Batch().v;
        }
        else if (orgParams.lookUpTableParams.lookUpTableAxis == LookUpTableAxis::X) {
            runInfo.gws0 = orgParams.inputs[0].Y().v;
            runInfo.gws1 = orgParams.inputs[0].Feature().v;
            runInfo.gws2 = orgParams.inputs[0].Batch().v;
        }

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        KernelData kd = KernelData::Default<LookUpTableParams>(params);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, "", false, false, 2);

        kd.estimatedTime = FORCE_PRIORITY_9;

        return{ kd };
    }
}