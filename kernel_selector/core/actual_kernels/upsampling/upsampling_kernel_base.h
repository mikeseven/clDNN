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

#include "common_kernel_base.h"

namespace kernel_selector
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // upsampling_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct upsampling_params : public BaseParams
    {
        upsampling_params() : BaseParams(KernelType::UPSAMPLING) {}

        struct DedicatedParams
        {
            uint32_t scale = 1;
            uint32_t num_filter = 0;
            SampleType sampleType = SampleType::NEAREST;
        };

        DedicatedParams usParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();
            k.EnableUpSamplingSampleType(usParams.sampleType);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // upsampling_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct upsampling_optional_params : OptionalParams
    {
        upsampling_optional_params() : OptionalParams(KernelType::UPSAMPLING) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // UpSamplingKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class UpSamplingKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~UpSamplingKernelBase() {}

        using DispatchData = CommonDispatchData;

    protected:
        virtual bool Validate(const Params& p, const OptionalParams& o) const override;
        virtual JitConstants GetJitConstants(const upsampling_params& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams& options) const;
    };
}