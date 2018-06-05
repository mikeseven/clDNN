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

#include "convolution_kernel_DPAS.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {
    
    ParamsKey ConvolutionKernel_DPAS::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableInputWeightsType(WeightsType::INT8);
        k.EnableInputLayout(DataLayout::byxf_af32);
        k.EnableOutputLayout(DataLayout::byxf_af32);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDepthwiseSeparableOpt();
        k.EnableInt8Quantization();
        k.EnableOutputCalibration();
        k.DisableTuning();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_DPAS::SetDefault(const ConvolutionParams& arg, int) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

        // Sub-group size used by "kernel_name_bfyx_os_iyx_osv16" kernel.
        constexpr size_t sub_group_size = 8;

        const auto of_maps = arg.output.Feature().v;
        const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);
        runInfo.cldnnStyle.leftovers = of_threads_per_batch - of_maps;

        const auto cp = arg.convParams;

        runInfo.effiency = FORCE_PRIORITY_3;

        runInfo.gws0 = arg.output.X().v;
        runInfo.gws1 = arg.output.Y().v;
        runInfo.gws2 = of_threads_per_batch * arg.output.Batch().v;

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = sub_group_size;

        return runInfo;
    }

    JitConstants ConvolutionKernel_DPAS::GetJitConstants(const ConvolutionParams& params, DispatchData runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws2));

        return jit;
    }

    KernelsData ConvolutionKernel_DPAS::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        KernelsData kd = GetCommonKernelsData(params, options);
        kd[0].estimatedTime = FORCE_PRIORITY_2;
        return kd;//return GetCommonKernelsData(params, options);
    }
}