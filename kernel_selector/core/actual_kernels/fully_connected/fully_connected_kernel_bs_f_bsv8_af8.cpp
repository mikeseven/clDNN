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

#include "fully_connected_kernel_bs_f_bsv8_af8.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{
    ParamsKey FullyConnected_bs_f_bsv8_af8::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputWeightsType(WeightsType::F16);
        k.SetInputWeightsType(WeightsType::F32);
        k.EnableAllInputLayout();
        //k.EnableAllOutputLayout();
        k.SetOutputLayout(DataLayout::fb);
        k.SetOutputLayout(DataLayout::bs_f_bsv8__af8);
        k.SetBatchingSupport();
        k.SetBiasPerOutput();
        k.SetNonBiasSupport();
        //k.SetOffsetSupport();
        //k.SetPitchesSupport();
        k.SetSubGroupSupport();
        return k;
    }

    FullyConnected_bs_f_bsv8_af8::DispatchData FullyConnected_bs_f_bsv8_af8::set_default(const FullyConnectedParams& arg) const
    {
        DispatchData kd = set_kernel_data(arg);

        size_t groups_per_batches = get_local_groups_size(arg);
        kd.gws0 = cldnn::align_to(arg.output.Length() / (get_neurons_per_work_item(arg) * get_batches_per_work_item(arg) * groups_per_batches), 8);
        kd.gws1 = groups_per_batches;
        kd.lws0 = 8;
        kd.lws1 = 1;

        kd.vload_kernel_type = true;

        return kd;
    }
    
    static bool check_input_layout(const DataTensor& t)
    {
        bool b16_layout = false;
        b16_layout |= t.layout == DataLayout::bs_f_bsv8__af8;
        b16_layout |= Tensor::channelndex(t.layout, Tensor::DataChannelName::NAME_BATCH) == 0 && (t.batch().v == 8); // TODO - check f alignment to 8
        return b16_layout;
    }

    static bool check_output_layout(const DataTensor& t)
    {
        bool b16_layout = false;
        b16_layout |= (t.layout == DataLayout::fb);
        b16_layout |= (t.layout == DataLayout::bs_f_bsv8__af8) && (t.batch().v == 8);
        return b16_layout;
    }

    KernelsData FullyConnected_bs_f_bsv8_af8::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        const auto& orgParams = static_cast<const FullyConnectedParams&>(params);
        const auto& orgOptParams = static_cast<const FullyConnectedOptionalParams&>(optParams);

        const bool bProperBatch = 
            orgParams.inputs[0].batch().v >= 8 &&
            orgParams.inputs[0].batch().v % 8 == 0;
        const bool bProperInput = check_input_layout(orgParams.inputs[0]);
        const bool bProperOutput = check_output_layout(orgParams.output);
        const bool bSupportedLayout = orgOptParams.allow_reorder_input || bProperInput;
        
        if (!bProperBatch || !bSupportedLayout || !bProperOutput)
        {
            return KernelsData();
        }
        
        return GetCommonKernelsData(params, optParams, DataLayout::bs_f_bsv8__af8, WeightsLayout::os_i_osv8__ai8, FORCE_PRIORITY_4);
    }
}