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

#include "fully_connected_kernel_fb_io_b8_f8.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{
    ParamsKey FullyConnected_fb_io_b8_f8::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F32);
        k.SetInputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetInputWeightsType(WeightsType::F32);
        k.SetInputWeightsType(WeightsType::F16);
        k.EnableAllInputLayout();
        k.SetOutputLayout(DataLayout::fb);
        k.SetBatchingSupport();
        k.SetBiasPerFeatureMap();
        k.SetNonBiasSupport();
        k.SetSubGroupSupport();
        return k;
    }

    FullyConnected_fb_io_b8_f8::DispatchData FullyConnected_fb_io_b8_f8::SetDefault(const FullyConnectedParams& arg) const
    {
        DispatchData kd = SetKernelData(arg);

        const auto& output = arg.output;
        
        size_t groups_per_batches = GetLocalGroupsSize(arg);
        kd.gws0 = output.Length() / (GetNeuronsPerWorkItem(arg) * GetBatchesPerWorkItem(arg) * groups_per_batches);
        kd.gws1 = groups_per_batches;
        kd.lws0 = 8;
        kd.lws1 = 1;

        kd.vload_kernel_type = true;

        return kd;
    }

    KernelsData FullyConnected_fb_io_b8_f8::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        const auto& orgParams = static_cast<const FullyConnectedParams&>(params);

        const auto& output = orgParams.output;
        const auto batches = output.Batch().v;
        const auto x_size = output.Length() / batches;


        const bool bSupportedBatch = (batches % 8) == 0;
        const bool bSupportedFeature = (x_size % 8) == 0;

        if (!bSupportedBatch || 
            !bSupportedFeature)
        {
            return KernelsData();
        }

        float estimated_time =
            orgParams.inputs[0].GetDType() == Datatype::F16 && batches >= 16 ?
            FORCE_PRIORITY_3 : FORCE_PRIORITY_5;
        
        return GetCommonKernelsData(params, optParams, DataLayout::fb, WeightsLayout::io, estimated_time);
    }
}