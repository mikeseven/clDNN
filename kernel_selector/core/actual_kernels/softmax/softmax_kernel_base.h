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
    // SoftMaxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct softmax_params : public BaseParams
    {
        softmax_params() : BaseParams(KernelType::SOFT_MAX) {}

        struct DedicatedParams
        {
            SoftmaxDim dim = SoftmaxDim::FEATURE;
        };

        DedicatedParams smParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();
            k.EnableSoftmaxDim(smParams.dim);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // softmax_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct softmax_optional_params : OptionalParams
    {
        softmax_optional_params() : OptionalParams(KernelType::SOFT_MAX) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftmaxKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class SoftmaxKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~SoftmaxKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {
            size_t itemsNum;
            size_t leftovers;
            size_t dataSetsCount;
            size_t dataSetSize;
            size_t normIndex; //which dimension (from in-memory representation) is normalized, e.g. for bfyx and softmax::normalize_f, it will be f's index == 2 (used only by naive kernel)
        };

    protected:
        virtual bool Validate(const Params&, const OptionalParams&) const;
        virtual JitConstants GetJitConstants(const softmax_params& params, DispatchData kd) const;
        virtual DispatchData SetDefault(const softmax_params& params, const OptionalParams& optParams) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams& optParams) const;
    };

    class SoftmaxKernelBaseBF : public SoftmaxKernelBase
    {
    public:
        using Parent = SoftmaxKernelBase;
        using Parent::Parent;
        virtual ~SoftmaxKernelBaseBF() {}

    protected:
        virtual bool Validate(const Params&, const OptionalParams&) const override;
        virtual DispatchData SetDefault(const softmax_params& params, const OptionalParams& optParams) const override;
    };
}