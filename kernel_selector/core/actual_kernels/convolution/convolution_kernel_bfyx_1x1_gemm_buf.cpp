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

#include "convolution_kernel_bfyx_1x1_gemm_buf.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {
    
    ParamsKey ConvolutionKernel_bfyx_1x1_gemm_buf::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputLayout(DataLayout::byxf);
        k.EnableOutputLayout(DataLayout::byxf);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableDepthwiseSeparableOpt();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_1x1_gemm_buf::SetDefault(const ConvolutionParams& params, int) const
    {
        DispatchData kd = ConvolutionKernelBase::SetDefault(params);

        const auto& out = params.output;

        auto x = out.X().v;
        auto y = out.Y().v;
        auto f = out.Feature().v;
        auto b = out.Batch().v;

        kd.gws0 = Align(f, 16);
        kd.gws1 = static_cast<size_t>(std::ceil(x*y / 16.0f));
        kd.gws2 = b;

        kd.lws0 = 16;
        kd.lws1 = 1;
        kd.lws2 = 1;

        kd.effiency = FORCE_PRIORITY_1;

        return kd;
    }

    bool ConvolutionKernel_bfyx_1x1_gemm_buf::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const ConvolutionParams&>(p);

        const auto &input = params.inputs[0];

        const bool bPad = input.X().pad.Total() != 0 || input.Y().pad.Total() != 0 || input.Feature().pad.Total() != 0 || input.Batch().pad.Total() != 0;
        const bool bFilterSize = params.convParams.filterSize.x != 1 || params.convParams.filterSize.y != 1;
        const bool bStride = params.convParams.stride.x != 1 || params.convParams.stride.y != 1;

        if(bPad || bFilterSize || bStride)
        {
            return false;
        }

        return true;
    }

    JitConstants ConvolutionKernel_bfyx_1x1_gemm_buf::GetJitConstants(const ConvolutionParams& params, DispatchData runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        const auto& out = params.output;
        const auto& input = params.inputs[0];

        auto x = out.X().v;
        auto y = out.Y().v;

        auto num_whole_groups_y = x*y / (16);
        auto num_whole_subgroups_y = (x*y - num_whole_groups_y*16) / 16;
        auto last_local_y = x*y - (num_whole_groups_y + num_whole_subgroups_y)*16;

        jit.AddConstant(MakeJitConstant("TX", 16));
        jit.AddConstant(MakeJitConstant("TY", 1));
        jit.AddConstant(MakeJitConstant("M", x*y));
        jit.AddConstant(MakeJitConstant("K", input.Feature().v));
        jit.AddConstant(MakeJitConstant("N", out.Feature().v));
        jit.AddConstant(MakeJitConstant("TILE_M", 16));
        jit.AddConstant(MakeJitConstant("TILE_N", 16));
        jit.AddConstant(MakeJitConstant("K8", (input.Feature().v >> 3)));
        jit.AddConstant(MakeJitConstant("NUM_WHOLE_GROUPS_Y", num_whole_groups_y));
        jit.AddConstant(MakeJitConstant("NUM_WHOLE_SUBGROUPS_Y", num_whole_subgroups_y));
        jit.AddConstant(MakeJitConstant("LAST_LOCAL_Y", last_local_y));

        return jit;
    }

    KernelsData ConvolutionKernel_bfyx_1x1_gemm_buf::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}