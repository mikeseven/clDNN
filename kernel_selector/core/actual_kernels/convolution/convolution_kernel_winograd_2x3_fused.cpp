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

#include "convolution_kernel_winograd_2x3_fused.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {

    ParamsKey ConvolutionKernel_Winograd_2x3_s1_fused::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();

        return k;
    }

    JitConstants ConvolutionKernel_Winograd_2x3_s1_fused::GetJitConstants(const ConvolutionParams& params, Parent::DispatchData runInfo) const
    {
        JitConstants jit = Parent::GetJitConstants(params, runInfo);

        const uint32_t idepth = (uint32_t)params.inputs[0].Feature().v;
        const uint32_t input_pad_y = +(uint32_t)params.inputs[0].Y().pad.before + (uint32_t)params.inputs[0].Y().pad.after;
        const uint32_t input_pad_x = +(uint32_t)params.inputs[0].X().pad.before + (uint32_t)params.inputs[0].X().pad.after;
        const uint32_t rows = (uint32_t)params.inputs[0].Y().v + input_pad_y;
        const uint32_t cols = (uint32_t)params.inputs[0].X().v + input_pad_x;

        uint32_t output_pad_x_before = (uint32_t)params.output.GetDims()[0].pad.before;
        uint32_t output_pad_y_before = (uint32_t)params.output.GetDims()[1].pad.before;
        uint32_t output_pad_x_after = (uint32_t)params.output.GetDims()[0].pad.after;
        uint32_t output_pad_y_after = (uint32_t)params.output.GetDims()[1].pad.after;
        uint32_t C4_up16 = ((uint32_t)((idepth + 15) / 16) * 16) / 4;

        jit.AddConstants({
            MakeJitConstant("H", rows),
            MakeJitConstant("W", cols),
            MakeJitConstant("P", rows - 3 + 1 + output_pad_y_before + output_pad_y_after),
            MakeJitConstant("Q", cols - 3 + 1 + output_pad_x_before + output_pad_x_after),
            MakeJitConstant("R", 3),
            MakeJitConstant("S", 3),
            MakeJitConstant("N", 1),
            MakeJitConstant("px", 0),
            MakeJitConstant("py", 0),
            MakeJitConstant("sx", 1),
            MakeJitConstant("sy", 1),

            MakeJitConstant("C4_up16", C4_up16),
            MakeJitConstant("TROWS", rows),
            MakeJitConstant("TCOLS", 4),
            MakeJitConstant("KROWSW", 3),
            MakeJitConstant("KCOLSW", 4),
        });

        return jit;
    }

    ConvolutionKernel_Winograd_2x3_s1_fused::Parent::DispatchData ConvolutionKernel_Winograd_2x3_s1_fused::SetDefault(const ConvolutionParams& arg, int) const
    {
        Parent::DispatchData runInfo = Parent::SetDefault(arg);

        const uint32_t odepth = (uint32_t)arg.output.Feature().v;
        const uint32_t input_pad_y = +(uint32_t)arg.inputs[0].Y().pad.before + (uint32_t)arg.inputs[0].Y().pad.after;
        const uint32_t input_pad_x = +(uint32_t)arg.inputs[0].X().pad.before + (uint32_t)arg.inputs[0].X().pad.after;
        const uint32_t rows = (uint32_t)arg.inputs[0].Y().v + input_pad_y;
        const uint32_t cols = (uint32_t)arg.inputs[0].X().v + input_pad_x;

        uint32_t P = rows - 2;
        uint32_t Q = cols - 2;
        uint32_t K = odepth;
        uint32_t N = 1;

        uint32_t global_step[3] = { 14, 4, 16 * 8 };
        uint32_t local_size[3] = { 8, 1, 8 };

        runInfo.gws0 = ((uint32_t)((Q + global_step[0] - 1)) / global_step[0]) * local_size[0];
        runInfo.gws1 = ((uint32_t)((P + global_step[1] - 1)) / global_step[1]) * local_size[1];
        runInfo.gws2 = ((uint32_t)((N*K * 8 + global_step[2] - 1)) / global_step[2]) * local_size[2];

        runInfo.lws0 = local_size[0];
        runInfo.lws1 = local_size[1];
        runInfo.lws2 = local_size[2];

        runInfo.effiency = FORCE_PRIORITY_1;

        return runInfo;
    }

    bool ConvolutionKernel_Winograd_2x3_s1_fused::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!Parent::Validate(p, o))
        {
            return false;
        }

        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);

        if (((uint32_t)params.weights.X().v != 3) || (uint32_t)(params.weights.Y().v != 3) ||
            ((uint32_t)params.convParams.stride.x != 1) ||
            ((uint32_t)params.convParams.stride.y != 1) ||
            ((uint32_t)params.convParams.filterSize.x != 3) ||
            ((uint32_t)params.convParams.filterSize.y != 3) ||
            ((uint32_t)params.convParams.padding.x > 1) ||
            ((uint32_t)params.convParams.padding.y > 1) ||
            ((uint32_t)params.output.Feature().v % 32) ||
            ((uint32_t)params.inputs[0].Feature().v % 32))
        {
            return{};
        }

        return true;
    }

    KernelsData ConvolutionKernel_Winograd_2x3_s1_fused::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}