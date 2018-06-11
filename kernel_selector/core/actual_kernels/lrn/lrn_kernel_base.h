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

#pragma once

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace kernel_selector 
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // lrn_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct lrn_params : public BaseParams
    {
        lrn_params() : BaseParams(KernelType::LRN), lrnParams() {}

        struct DedicatedParams
        {
            LRNMode             normMode = LRNMode::ACROSS_CHANNEL;
            KernelDividerMode   divMode = KernelDividerMode::DONT_CARE;
            float               alpha = 0.f;
            float               beta = 0.f;
            float               k = 0.f;
            uint32_t            localSize = 0;
        };

        DedicatedParams lrnParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.EnableLRNMode(lrnParams.normMode);
            k.EnableLRNKernelDividerMode(lrnParams.divMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // lrn_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct lrn_optional_params : OptionalParams
    {
        lrn_optional_params() : OptionalParams(KernelType::LRN) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // lrn_kernel_base
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class LRNKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~LRNKernelBase() {}

        using DispatchData = CommonDispatchData;

    protected:
        virtual bool Validate(const Params& p, const OptionalParams& o) const override;
        virtual JitConstants GetJitConstants(const lrn_params& params, DispatchData kd) const;
        virtual DispatchData SetDefault(const lrn_params& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams&, float estimatedTime) const;
    };
}