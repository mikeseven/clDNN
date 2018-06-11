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

#include "reorder_weights_winograd_2x3_kernel.h"
#include "kernel_selector_utils.h"
 
namespace kernel_selector 
{
    ParamsKey ReorderWeightsWinograd2x3Kernel::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputWeightsType(WeightsType::F16);
        k.EnableOutputWeightsType(WeightsType::F32);
        k.EnableAllInputWeightsLayout();
        k.EnableOutputWeightsLayout(WeightsLayout::winograd_2x3_s1_weights);
        k.EnableOutputWeightsLayout(WeightsLayout::winograd_2x3_s1_fused_weights);
        k.EnableWinogradReorder();
        k.EnableDifferentTypes();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        return k;
    }

    ReorderWeightsWinograd2x3Kernel::DispatchData ReorderWeightsWinograd2x3Kernel::SetDefault(const ReorderWeightsParams& params) const
    {
        DispatchData kd;

        const auto& input = params.reorderParams.input;

        kd.gws0 = 1;
        kd.gws1 = 3;
        kd.gws2 = static_cast<size_t>(input.IFM().v * input.OFM().v);

        kd.lws0 = 1;
        kd.lws1 = 1;
        kd.lws2 = 32;

        return kd;
    }

    KernelsData ReorderWeightsWinograd2x3Kernel::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        const ReorderWeightsParams& orgParams = static_cast<const ReorderWeightsParams&>(params);
        return GetCommonKernelsData(orgParams, options, FORCE_PRIORITY_4);
    }
}