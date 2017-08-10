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

#include "pooling_kernel_gpu_bfyx_block_opt.h"
 
namespace KernelSelector 
{
    ParamsKey PoolingKernelGPUBfyxBlockOpt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnablePoolType(PoolType::MAX);
        k.EnablePoolType(PoolType::AVG);
        k.EnablePoolRemainder(PoolRemainder::FLOOR);
        k.EnablePoolRemainder(PoolRemainder::CEIL);
        k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
        return k;
    }

    PoolingKernelBase::DispatchData PoolingKernelGPUBfyxBlockOpt::SetDefault(const PoolingParams& params) const
    {
        const auto& output = params.output;

        DispatchData runInfo = PoolingKernelBase::SetDefault(params);

        runInfo.gws1 = Align(output.Y().v, params.poolParams.poolSize.y) / params.poolParams.poolSize.y;

        return runInfo;
    }

    JitConstants PoolingKernelGPUBfyxBlockOpt::GetJitConstants(const PoolingParams& params, DispatchData kd) const
    {
        auto mem_consts = PoolingKernelBase::GetJitConstants(params, kd);

        mem_consts.AddConstant(MakeJitConstant("BLOCK_SIZE_Y", params.poolParams.poolSize.y + params.poolParams.poolSize.y*params.poolParams.poolStride.y - 1));

        return mem_consts;
    }

    bool PoolingKernelGPUBfyxBlockOpt::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!PoolingKernelBase::Validate(p, o))
        {
            return false;
        }

        const PoolingParams& params = static_cast<const PoolingParams&>(p);
        if (NeedsBoundaryCheck(params) ||
            params.poolParams.poolSize.x > 5 || params.poolParams.poolSize.y > 5)
        {
            return false;
        }

        return true;
    }

    KernelsData PoolingKernelGPUBfyxBlockOpt::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const PoolingParams& orgParams = static_cast<const PoolingParams&>(params);

        DispatchData runInfo = SetDefault(orgParams);

        KernelData kd = KernelData::Default<PoolingParams>(params);

        auto cldnn_jit = GetJitConstants(orgParams, runInfo);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);

        kd.estimatedTime = FORCE_PRIORITY_8;

        return{ kd };
    }
}