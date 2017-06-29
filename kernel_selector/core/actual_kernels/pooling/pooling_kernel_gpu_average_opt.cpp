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

#include "pooling_kernel_gpu_average_opt.h"
 
namespace KernelSelector 
{
    ParamsKey PoolingKernelGPUAverageOpt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnablePoolType(PoolType::AVG);
        k.EnablePoolRemainder(PoolRemainder::FLOOR);
        k.EnablePoolRemainder(PoolRemainder::CEIL);
        k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
        return k;
    }

    PoolingKernelBase::DispatchData PoolingKernelGPUAverageOpt::SetDefault(const PoolingParams& params) const
    {
        DispatchData runInfo = PoolingKernelBase::SetDefault(params);

        const int simdSize = 16;
        runInfo.tileWidth = simdSize - 2;
        runInfo.tileHeight = 7;

        const int numTilesX = static_cast<int>(std::ceil(static_cast<float>(params.inputs[0].X().v) / static_cast<float>(runInfo.tileWidth)));
        const int numTilesY = static_cast<int>(std::ceil(static_cast<float>(params.inputs[0].Y().v) / static_cast<float>(runInfo.tileHeight)));

        runInfo.gws0 = numTilesX * simdSize;
        runInfo.gws1 = numTilesY;
        runInfo.gws2 = params.inputs[0].Feature().v;
        runInfo.lws0 = simdSize;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        return runInfo;
    }

    KernelsData PoolingKernelGPUAverageOpt::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::POOLING);

        const PoolingParams& orgParams = static_cast<const PoolingParams&>(params);

        if (orgParams.activationFunc != ActivationFunction::NONE)
        {
            return{};
        }

        if ((orgParams.poolParams.poolSize.x != 3) ||
            (orgParams.poolParams.poolSize.y != 3) ||
            (orgParams.poolParams.poolStride.x != 1) ||
            (orgParams.poolParams.poolStride.y != 1) ||
            (orgParams.poolParams.poolPad.x != 1) || 
            (orgParams.poolParams.poolPad.y != 1) ||
            !(orgParams.inputs[0] == orgParams.output) ||
            orgParams.inputs[0].PaddingExists() ||
            orgParams.output.PaddingExists())
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

        kd.estimatedTime = FORCE_PRIORITY_8;

        return{ kd };
    }
}