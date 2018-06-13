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
    // reorder_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct reorder_params : public BaseParams
    {
        reorder_params() : BaseParams(KernelType::REORDER) {}

        struct DedicatedParams
        {
            MeanSubtractMode    mode = MeanSubtractMode::NONE;
            MeanOp              mean_op = MeanOp::SUB;
            std::vector<float>  meanValues;
            DataTensor          mean;
            uint32_t            winograd_input_offset_x;
            uint32_t            winograd_input_offset_y;
            uint32_t            winograd_nr_tiles_x;
            bool                winograd = false;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();

            if (reorderParams.winograd)
            {
                k.EnableWinogradReorder();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // reorder_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct reorder_optional_params : OptionalParams
    {
        reorder_optional_params() : OptionalParams(KernelType::REORDER) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // reorder_weights_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct reorder_weights_params : public Params
    {
        reorder_weights_params() : Params(KernelType::REORDER, ""), reorderParams() {}

        struct DedicatedParams
        {
            WeightsTensor input;
            WeightsTensor output;
            bool winograd = false;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k;
            const auto& input = reorderParams.input;
            const auto& output = reorderParams.output;
            k.EnableInputWeightsType(input.GetDType());
            k.EnableOutputWeightsType(output.GetDType());
            k.EnableInputWeightsLayout(input.GetLayout());
            k.EnableOutputWeightsLayout(output.GetLayout());

            if (input.PitchesDifferFromLogicalDims() ||
                output.PitchesDifferFromLogicalDims())
            {
                k.EnableTensorPitches();
            }

            if (input.GetFirstElementOffset() != 0 || output.GetFirstElementOffset() != 0)
            {
                k.EnableTensorOffset();
            }

            if (reorderParams.winograd)
            {
                k.EnableWinogradReorder();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ReorderKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~ReorderKernelBase() {}

        using DispatchData = CommonDispatchData;
    
    protected:
        virtual JitConstants GetJitConstants(const reorder_weights_params& params) const;
        virtual JitConstants GetJitConstants(const reorder_params& params) const;
        virtual DispatchData SetDefault(const reorder_weights_params& params) const;
        virtual DispatchData SetDefault(const reorder_params& params) const;
        KernelsData GetCommonKernelsData(const reorder_weights_params& params, const OptionalParams&, float estimated_time) const;
        KernelsData GetCommonKernelsData(const reorder_params& params, const OptionalParams&, float estimated_time) const;
    };
}