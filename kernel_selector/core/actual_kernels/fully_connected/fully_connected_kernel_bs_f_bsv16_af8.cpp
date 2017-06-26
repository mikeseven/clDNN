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

#include "fully_connected_kernel_bs_f_bsv16_af8.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{
    ParamsKey FullyConnected_bs_f_bsv16_af8::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        //k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        //k.SetOutputDataType(Datatype::F32);
        k.SetInputWeightsType(WeightsType::F16);
        //k.SetInputWeightsType(WeightsType::F32);
        k.EnableAllInputLayout();
        //k.EnableAllOutputLayout();
        k.SetOutputLayout(DataLayout::fb);
        k.SetOutputLayout(DataLayout::bs_f_bsv16__af8);
        k.SetBatchingSupport();
        k.SetBiasPerFeatureMap();
        k.SetNonBiasSupport();
        //k.SetOffsetSupport();
        //k.SetPitchesSupport();
        k.SetSubGroupSupport();
        return k;
    }

    FullyConnected_bs_f_bsv16_af8::DispatchData FullyConnected_bs_f_bsv16_af8::SetDefault(const FullyConnectedParams& arg) const
    {
        DispatchData kd = FullyConnectedKernelBase::SetDefault(arg);

        size_t groups_per_batches = GetLocalGroupsSize(arg);
        kd.gws0 = cldnn::align_to(arg.output.Length() / (GetBatchesPerWorkItem(arg) * groups_per_batches), 16);
        kd.gws1 = groups_per_batches;
        kd.lws0 = 16;
        kd.lws1 = 1;

        kd.vload_kernel_type = true;

        return kd;
    }
    
    static bool check_input_layout(const DataTensor& t)
    {
        bool b16_layout = false;
        b16_layout |= t.GetLayout() == DataLayout::bs_f_bsv16__af8;
        b16_layout |= Tensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::BATCH) == 0 && t.Batch().v == 16;
        return b16_layout;
    }

    bool FullyConnected_bs_f_bsv16_af8::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!FullyConnectedKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const FullyConnectedParams&>(p);
        const auto& optParams = static_cast<const FullyConnectedOptionalParams&>(o);

        const bool bProperBatch = params.inputs[0].Batch().v == 16;
        const bool bProperInput = check_input_layout(params.inputs[0]);
        const bool bSupportedLayout = optParams.allowReorderInput || bProperInput;

        if (!bProperBatch || !bSupportedLayout)
        {
            return false;
        }

        return true;
    }

    KernelsData FullyConnected_bs_f_bsv16_af8::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {   
        return GetCommonKernelsData(params, optParams, DataLayout::bs_f_bsv16__af8, { WeightsLayout::os_i_osv16__ai8 }, FORCE_PRIORITY_2);
    }
}