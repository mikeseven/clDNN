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

#include "deconvolution_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{

    ParamsKey DeconvolutionKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::yxfb);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableInputLayout(DataLayout::byxf);
        k.EnableOutputLayout(DataLayout::yxfb);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::byxf);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDepthwiseSeparableOpt();
        return k;
    }

    CommonDispatchData DeconvolutionKernelRef::SetDefault(const DeconvolutionParams& params) const
    {
        CommonDispatchData runInfo = DeconvolutionKernelBase::SetDefault(params);

        if (params.output.Feature().v * params.output.Batch().v <= 16)
        {
            const auto& out = params.output;
            runInfo.gws0 = Align(out.X().v, 32);
            runInfo.gws1 = out.Y().v;
            runInfo.gws2 = out.Feature().v * out.Batch().v;

            runInfo.lws0 = 32;
            runInfo.lws1 = 1;
            runInfo.lws2 = 1;
        }

        return runInfo;
    }

    JitConstants DeconvolutionKernelRef::GetJitConstants(const DeconvolutionParams& params) const
    {
        auto jit = DeconvolutionKernelBase::GetJitConstants(params);

        if (params.output.Feature().v * params.output.Batch().v <= 16)
            jit.AddConstant(MakeJitConstant("DIM_ORDER_XYBF", 1));

        return jit;
    }

    KernelsData DeconvolutionKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::DECONVOLUTION);

        const DeconvolutionParams& orgParams = static_cast<const DeconvolutionParams&>(params);

        const std::vector<WeightsLayout> weightsLayouts = {
            WeightsLayout::yxio,
            WeightsLayout::iyxo,
            WeightsLayout::oyxi,
            WeightsLayout::oiyx };

        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<DeconvolutionParams>(params);
        DeconvolutionParams& newParams = *static_cast<DeconvolutionParams*>(kd.params.get());

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            weightsLayouts,
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, ROUND_ROBIN, true, !orgParams.bias.empty());
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}