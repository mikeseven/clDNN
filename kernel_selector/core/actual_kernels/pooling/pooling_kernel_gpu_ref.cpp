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

#include "pooling_kernel_gpu_ref.h"
 
namespace KernelSelector 
{
    ParamsKey PoolingKernelGPURef::GetSupportedKey() const
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
        k.SetPoolType(PoolType::MAX);
        k.SetPoolType(PoolType::AVG);
        k.SetPoolRemainder(PoolRemainder::FLOOR);
        k.SetPoolRemainder(PoolRemainder::CEIL);
        k.SetPoolKernelDividerMode(KernelDividerMode::FIXED);
        k.SetPoolKernelDividerMode(KernelDividerMode::DYNAMIC);
        return k;
    }

    KernelsData PoolingKernelGPURef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::POOLING);

        const PoolingParams& orgParams = static_cast<const PoolingParams&>(params);

        if (orgParams.activationFunc != ActivationFunction::NONE)
        {
            return{};
        }
        
        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<PoolingParams>(params);

        auto cldnn_jit = GetJitConstants(orgParams, runInfo);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);

        kd.estimatedTime = FORCE_PRIORITY_9;

        return{ kd };
    }
}