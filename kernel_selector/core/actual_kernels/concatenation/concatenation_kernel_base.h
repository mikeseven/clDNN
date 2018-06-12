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
    // concatenation_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct concatenation_params : public BaseParams
    {
        concatenation_params() : BaseParams(KernelType::CONCATENATION), concatParams() {}

        struct DedicatedParams
        {
            ConcatAxis axis = ConcatAxis::FEATURE;
        };

        DedicatedParams concatParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();
            k.EnableConcatAxis(concatParams.axis);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // concatenation_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct concatenation_optional_params : OptionalParams
    {
        concatenation_optional_params() : OptionalParams(KernelType::CONCATENATION) {}
        bool kernelPerInput = true;

        virtual ParamsKey GetSupportedKey() const
        {
            ParamsKey k = OptionalParams::GetSupportedKey();

            if (kernelPerInput)
            {
                k.EnableConcatKernelPerInput();
            }
            else
            {
                k.EnableConcatOneKernel();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConcatenationKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ConcatenationKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~ConcatenationKernelBase() {}

        using DispatchData = CommonDispatchData;
    
    protected:
        virtual bool Validate(const Params&, const OptionalParams&) const override;
        virtual JitConstants GetJitConstants(const concatenation_params& params) const;
        virtual DispatchData SetDefault(const concatenation_params& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams&) const;
    };
}