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

#include "pooling_kernel_base.h"

namespace KernelSelector 
{
    bool PoolingKernelBase::Validate(const Params& p, const OptionalParams& o) const
    {
        if (p.GetType() != KernelType::POOLING ||
            o.GetType() != KernelType::POOLING)
        {
            return false;
        }

        return true;
    }

    JitConstants PoolingKernelBase::GetJitConstants(const PoolingParams& params, PoolingKernelBase::DispatchData kd) const
    {
        JitConstants mem_consts = MakePoolingJitConstants(params);

        if (kd.needsBoundary)
        {
            mem_consts.AddConstant(MakeJitConstant("CHECK_BOUNDRY", 1));
        }

        return mem_consts;
    }

    // Checks if we need boundary checking in kernel.
    static bool NeedsBoundaryCheck(const PoolingParams& params)
    {
        const auto& pp = params.poolParams;

        if (pp.poolPad.x != 0 || pp.poolPad.y != 0)
        {
            return true;
        }

        const auto& input = params.inputs[0];

        if (input.X().v < pp.poolSize.x || input.Y().v < pp.poolSize.y)
        {
            return true;
        }

        auto mod_x = (input.X().v - pp.poolSize.x) % pp.poolStride.x;
        auto mod_y = (input.Y().v - pp.poolSize.y) % pp.poolStride.y;

        return mod_x || mod_y;
    }

    PoolingKernelBase::DispatchData PoolingKernelBase::SetDefault(const PoolingParams& params) const
    {
        const auto& output = params.output;

        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        if (output.GetLayout() == DataLayout::bfyx)
        {
            // Determine global work sizes.
            kd.gws2 = output.Batch().v * output.Feature().v;    // B, F
            kd.gws0 = Align(output.X().v, 32);        // X
            kd.gws1 = output.Y().v;                             // Y

            // Find largest positive local work size that is divider for global work size.
            kd.lws0 = 32;
            kd.lws1 = 1;
            kd.lws2 = 1;
        }
        else
        {
            // Determine global work sizes.
            kd.gws0 = output.Batch().v * output.Feature().v;    // B, F
            kd.gws1 = output.X().v;                             // X
            kd.gws2 = output.Y().v;                             // Y

            kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
            while (kd.gws0 % kd.lws0 != 0)
            {
                --kd.lws0;
            }
            kd.lws1 = 1;
            kd.lws2 = 1;
        }

        kd.needsBoundary = NeedsBoundaryCheck(params);

        return kd;
    }
}