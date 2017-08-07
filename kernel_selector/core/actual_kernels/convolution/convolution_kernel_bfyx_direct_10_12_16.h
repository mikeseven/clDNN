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

#pragma once

#include "convolution_kernel_base.h"
 
namespace KernelSelector {
    
    class ConvolutionKernel_bfyx_Direct_10_10_12 : public ConvolutionKernelBase
    {
    public:
        using Parent = ConvolutionKernelBase;
        ConvolutionKernel_bfyx_Direct_10_10_12() : ConvolutionKernelBase("convolution_gpu_bfyx_direct_10_12_16") {}
        virtual ~ConvolutionKernel_bfyx_Direct_10_10_12() {}

        virtual KernelsData GetKernelsData(const Params& params, const OptionalParams& options) const override;
        virtual ParamsKey GetSupportedKey() const override;

    protected:
        virtual std::vector<WeightsLayout> GetSupportedWeightLayouts() const override { return{ WeightsLayout::i_yxs_os_yxsv2_osv16 }; }

        virtual JitConstants GetJitConstants(const ConvolutionParams& params, DispatchData kd) const override;
        bool Validate(const Params& p, const OptionalParams& o) const override;
        DispatchData SetDefault(const ConvolutionParams& arg) const override;
    };
}