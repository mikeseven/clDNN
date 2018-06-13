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

namespace kernel_selector
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // eltwise_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct eltwise_params : public base_params
    {
        eltwise_params() : base_params(KernelType::ELTWISE), eltwiseParams() {}

        struct InputType
        {
            EltwiseInputMode mode = EltwiseInputMode::INPUT_BUFFER;
            uint32_t         index = 0;    // for inputs results;
            uint32_t         tmpIndex = 0;    // for temp results;
            float            scalar = 0.f;

            static InputType Buffer(uint32_t index)
            {
                eltwise_params::InputType input;
                input.mode = EltwiseInputMode::INPUT_BUFFER;
                input.index = index;
                return input;
            }

            static InputType UnorderedAccessBuffer(uint32_t index, uint32_t tmpIndex)
            {
                eltwise_params::InputType input;
                input.mode = EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER;
                input.index = index;
                input.tmpIndex = tmpIndex;
                return input;
            }

            static InputType Intermediate(uint32_t tmpIndex)
            {
                eltwise_params::InputType input;
                input.mode = EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX;
                input.tmpIndex = tmpIndex;
                return input;
            }

            static InputType Scalar(float s)
            {
                eltwise_params::InputType input;
                input.mode = EltwiseInputMode::SCALAR;
                input.scalar = s;
                return input;
            }

            static InputType OutBuffer()
            {
                eltwise_params::InputType output;
                output.mode = EltwiseInputMode::OUTPUT_BUFFER;
                return output;
            }
        };

        struct Node
        {
            std::vector<InputType> inputs;
            EltwiseMode mode;
        };

        struct UpdateInputData
        {
            uint32_t inputId;
            uint32_t tmpId;
        };

        struct DedicatedParams
        {
            std::vector<eltwise_params::Node> operations;
            std::vector<UpdateInputData> updateInputIds;
            bool layoutBased = false;
        };

        DedicatedParams eltwiseParams;

        virtual ParamsKey GetParamsKey() const
        {
            return base_params::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // eltwise_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct eltwise_optional_params : optional_params
    {
        eltwise_optional_params() : optional_params(KernelType::ELTWISE) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EltwiseKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class EltwiseKernelBase : public common_kernel_base
    {
    public:
        using common_kernel_base::common_kernel_base;
        virtual ~EltwiseKernelBase() {}

        using DispatchData = CommonDispatchData;
        JitConstants GetJitConstantsCommon(const eltwise_params& params, bool useVload8) const;

    protected:
        virtual bool Validate(const Params& p, const optional_params& o) const override;
        virtual JitConstants GetJitConstants(const eltwise_params& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const optional_params& options) const;
    };
}