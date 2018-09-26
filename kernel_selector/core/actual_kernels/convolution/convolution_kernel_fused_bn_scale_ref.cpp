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

#include "convolution_kernel_fused_bn_scale_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
    
    ParamsKey convolution_kernel_fused_bn_scale_ref::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableSplitSupport();
        k.EnableBatching();
        k.EnableOutputCalibration();
        k.EnableFusedBNScale();
        k.DisableTuning();
        return k;
    }

    ConvolutionKernelBase::DispatchData convolution_kernel_fused_bn_scale_ref::SetDefault(const convolution_params& arg, int) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

        runInfo.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        runInfo.gws0 = arg.output.Batch().v;
        runInfo.gws1 = arg.output.Feature().v; 
        runInfo.gws2 = 1;

        runInfo.lws0 = std::min(std::max(runInfo.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (runInfo.gws0 % runInfo.lws0 != 0)
        {
            --runInfo.lws0;
        }
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        return runInfo;
    }

    JitConstants convolution_kernel_fused_bn_scale_ref::GetJitConstants(const convolution_params& params, const DispatchData& runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        if (params.fused_in_training)
            jit.AddConstant(MakeJitConstant("FUSED_TRAINING", 1));
        if (params.scale_bias)
            jit.AddConstant(MakeJitConstant("SCALE_BIAS_TERM", 1));
        jit.AddConstant(MakeJitConstant("EPSILON", params.epsilon));

        return jit;
    }

    KernelsData convolution_kernel_fused_bn_scale_ref::GetKernelsData(const Params& params, const optional_params& options) const
    {
        KernelsData kd = GetCommonKernelsData(params, options);
        if(!kd.empty())
            kd[0].estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        auto& conv_params = static_cast<const convolution_params&>(params);
        auto& kernel = kd[0].kernels[0];

        kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 1 });
        if (conv_params.scale_bias)
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 2 });
        if (conv_params.fused_in_training)
        {
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 3 });
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 4 });
            kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 5 });
        }
 
        return kd;
    }
}