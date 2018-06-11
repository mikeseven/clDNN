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

#pragma once

#include "weight_bias_kernel_base.h"
#include "kernel_selector_params.h"

namespace KernelSelector 
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionGradWeightsParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionGradWeightsParams : public WeightBiasParams
    {
        ConvolutionGradWeightsParams() : WeightBiasParams(KernelType::CONVOLUTION_GRAD_WEIGHTS), convGradWeightsParams() {}

        struct DedicatedParams
        {
            uSize    filterSize;
            uSize    stride;
            uSize    dilation;
            uSize    padding;
            uint32_t split = 1;
            bool     depthwiseSeparableOpt = false;
        };

        DedicatedParams convGradWeightsParams;

        virtual std::string to_string() const override;

        virtual ParamsKey GetParamsKey() const override
        {
            ParamsKey k = WeightBiasParams::GetParamsKey();

            if (convGradWeightsParams.split > 1)
            {
                k.EnableSplitSupport();
            }

            if (convGradWeightsParams.dilation.x != 1 ||
                convGradWeightsParams.dilation.y != 1)
            {
                k.EnableDilation();
            }

            if (convGradWeightsParams.depthwiseSeparableOpt)
            {
                k.EnableDepthwiseSeparableOpt();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutiongradWeightsOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutiongradWeightsOptionalParams : WeightsBiasOptionalParams
    {
        ConvolutiongradWeightsOptionalParams() : WeightsBiasOptionalParams(KernelType::CONVOLUTION_GRAD_WEIGHTS) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionGradWeightsKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ConvolutionGradWeightsKernelBase : public WeightBiasKernelBase
    {
    public:
        using WeightBiasKernelBase::WeightBiasKernelBase;
        virtual ~ConvolutionGradWeightsKernelBase() {}

        using DispatchData = CommonDispatchData;
    
    protected:
        virtual KernelsData GetKernelsData(const Params& params, const OptionalParams& options) const;
        virtual JitConstants GetJitConstants(const ConvolutionGradWeightsParams& params) const;
        virtual DispatchData SetDefault(const ConvolutionGradWeightsParams& params) const;
    };
}