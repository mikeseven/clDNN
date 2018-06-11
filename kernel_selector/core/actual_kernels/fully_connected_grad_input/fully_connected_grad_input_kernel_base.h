﻿/*
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

namespace kernel_selector 
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedGradInputParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct FullyConnectedGradInputParams : public WeightBiasParams
    {
        FullyConnectedGradInputParams() : WeightBiasParams(KernelType::FULLY_CONNECTED_GRAD_INPUT) {}

        virtual ParamsKey GetParamsKey() const
        {
            return WeightBiasParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedGradInputOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct FullyConnectedGradInputOptionalParams : WeightsBiasOptionalParams
    {
        FullyConnectedGradInputOptionalParams() : WeightsBiasOptionalParams(KernelType::FULLY_CONNECTED_GRAD_INPUT) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedGradInputKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class FullyConnectedGradInputKernelBase : public WeightBiasKernelBase
    {
    public:
        using WeightBiasKernelBase::WeightBiasKernelBase;
        virtual ~FullyConnectedGradInputKernelBase() {}

        using DispatchData = CommonDispatchData;
    
    protected:
        virtual KernelsData GetKernelsData(const Params& params, const OptionalParams& options) const;
        virtual JitConstants GetJitConstants(const FullyConnectedGradInputParams& params) const;
        virtual DispatchData SetDefault(const FullyConnectedGradInputParams& params) const;
    };
}