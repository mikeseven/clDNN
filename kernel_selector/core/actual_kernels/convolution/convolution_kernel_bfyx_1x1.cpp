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

#include "convolution_kernel_bfyx_1x1.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {
    
    ParamsKey ConvolutionKernel_bfyx_1x1::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableInputLayout(DataLayout::bf8_xy16);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDepthwiseSeparableOpt();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_1x1::SetDefault(const ConvolutionParams& params, int) const
    {
        DispatchData kd = ConvolutionKernelBase::SetDefault(params);

        const auto& out = params.output;

        std::vector<size_t> global = { out.X().v, out.Y().v, out.Feature().v*out.Batch().v };
        auto local = GetOptimalLocalWorkGroupSizes(global);

        auto x = out.X().v;
        auto y = out.Y().v;
        auto f = out.Feature().v;
        auto b = out.Batch().v;

        kd.gws0 = Align(x * y, 16) / 16;
        kd.gws1 = f;
        kd.gws2 = b;

        kd.lws0 = 1;
        kd.lws1 = 16;
        kd.lws2 = 1;

        kd.effiency = FORCE_PRIORITY_3;

        return kd;
    }

    bool ConvolutionKernel_bfyx_1x1::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const ConvolutionParams&>(p);

        const auto &input = params.inputs[0];
        const auto &output = params.output;

        if(output.X().v != input.X().v || output.Y().v != input.Y().v)
        {
            return false;
        }
        if (input.X().pad.Total() != 0 || input.Y().pad.Total() != 0 || input.Feature().pad.Total() != 0 || output.Batch().pad.Total() != 0)
        {
            return false;
        }
        return true;
    }

    JitConstants ConvolutionKernel_bfyx_1x1::GetJitConstants(const ConvolutionParams& params, DispatchData runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        const auto& in = params.inputs[0];

        jit.AddConstant(MakeJitConstant("SIMDS_PER_OFM", (in.Feature().v % 32 == 0) ? 2 : 1));

        return jit;
    }

    KernelsData ConvolutionKernel_bfyx_1x1::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}