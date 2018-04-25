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

#include "pooling_kernel_gpu_byxf_padding_opt.h"

namespace KernelSelector
{
    ParamsKey PoolingKernelGPUByxfPaddingOpt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::byxf);
        k.EnableOutputLayout(DataLayout::byxf);
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

    PoolingKernelBase::DispatchData PoolingKernelGPUByxfPaddingOpt::SetDefault(const PoolingParams& params) const
    {
        const auto& output = params.output;

        DispatchData runInfo = PoolingKernelBase::SetDefault(params);

        runInfo.gws2 = output.Batch().v * (CeilDiv(output.Feature().v, 8));

        return runInfo;
    }

    JitConstants PoolingKernelGPUByxfPaddingOpt::GetJitConstants(const PoolingParams& params, DispatchData kd) const
    {
        auto mem_consts = PoolingKernelBase::GetJitConstants(params, kd);

        return mem_consts;
    }

    bool PoolingKernelGPUByxfPaddingOpt::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!PoolingKernelBase::Validate(p, o))
        {
            return false;
        }

        const PoolingParams& params = static_cast<const PoolingParams&>(p);
        if (params.inputs[0].Feature().v % 8 != 0)
            return false;

        return true;
    }

    KernelsData PoolingKernelGPUByxfPaddingOpt::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options, FORCE_PRIORITY_8);
    }
}