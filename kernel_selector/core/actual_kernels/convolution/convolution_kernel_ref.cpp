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

#include "convolution_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {
    
    ParamsKey ConvolutionKernelRef::GetSupportedKey() const
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
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        return k;
    }

    KernelsData ConvolutionKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);

        const bool bSupportedWeightsLayout = orgParams.weights.GetLayout() == WeightsLayout::oiyx;
        const bool bWeightsOK = bSupportedWeightsLayout || optParams.allowWeightsReorder;

        if (!bWeightsOK)
        {
            return{};
        }

        KernelData kd = KernelData::Default<ConvolutionParams>(params, 1);

        ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());
        const std::string kernel_id = params.layerID + std::to_string(UniqeID());

        SubGroupInfo runInfo;
        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << GetConvolutionJit(newParams, runInfo);

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { out.X().v, out.Y().v, out.Feature().v*out.Batch().v };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.argsDesc = GetArgumentDesc(1, true, !newParams.bias.empty());
        kernel.argsDesc.data.push_back({ ArgumentDescpirtor::Types::SPLIT, 0 });

        bool succeed = SetWeightsReorderParams(newParams, WeightsLayout::oiyx, kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}