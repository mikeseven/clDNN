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

#include "convolution_kernel_yxfb_yxio_b1_block.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b1_block::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::yxfb);
        k.EnableOutputLayout(DataLayout::yxfb);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDilation();
        k.EnableSubGroup();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b1_block::SetDefault(const ConvolutionParams& arg) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);
        // TODO: fill the proper data here (I don't know where can I locate it).
        return runInfo;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b1_block::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        
        if (!CheckPitchForSplitOnly(orgParams))
        {
            return{};
        }

        DispatchData runInfo = SetDefault(orgParams);

        KernelData kd = KernelData::Default<ConvolutionParams>(params);
        ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            { WeightsLayout::yxio },
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto cldnn_jit = GetJitConstants(orgParams, runInfo);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, true, !orgParams.bias.empty());
        kernel.argsDesc.data.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}