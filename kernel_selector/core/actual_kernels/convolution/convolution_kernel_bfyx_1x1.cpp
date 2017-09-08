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
        k.EnableInputLayout(DataLayout::f8_xy16);
        k.EnableOutputLayout(DataLayout::f8_xy16);
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

    ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_1x1::SetDefault(const ConvolutionParams& params) const
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

        if(kd.gws1 % 32 == 0)
        {
            kd.lws1 = 32;
        }
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

    KernelsData ConvolutionKernel_bfyx_1x1::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);

        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<ConvolutionParams>(params);
        ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            GetSupportedWeightLayouts(),
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto cldnn_jit = GetJitConstants(newParams, runInfo);
        
        const auto& in = orgParams.inputs[0];

        cldnn_jit.AddConstant(MakeJitConstant("SIMDS_PER_OFM", (in.Feature().v % 32 == 0) ? 2 : 1));

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, ROUND_ROBIN, true, !newParams.bias.empty());
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;
        if((orgParams.weights.X().v == 1) && (orgParams.weights.Y().v == 1))
        {
            kd.estimatedTime = FORCE_PRIORITY_9;
        }
        return{ kd };
    }
}