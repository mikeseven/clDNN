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

#include "fully_connected_kernel_fb_oi_b8_ref.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{
    ParamsKey FullyConnected_fb_oi_b8_ref::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputWeightsType(WeightsType::F32);
        k.EnableAllInputLayout();
        k.SetOutputLayout(DataLayout::fb);
        k.SetBatchingSupport();
        k.SetBiasPerFeatureMap();
        k.SetNonBiasSupport();
        k.SetSubGroupSupport();
        return k;
    }

    FullyConnected_fb_oi_b8_ref::DispatchData FullyConnected_fb_oi_b8_ref::set_default(const FullyConnectedParams& arg) const
    {
        DispatchData kd = set_kernel_data(arg);

        const auto& output = arg.output;
        kd.gws0 = output.batch().v;
        kd.gws1 = output.Length() / kd.gws0;
        kd.lws0 = 8;
        kd.lws1 = 1;

        return kd;
    }

    KernelsData FullyConnected_fb_oi_b8_ref::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        const auto& orgParams = static_cast<const FullyConnectedParams&>(params);

        const bool bSupportedBatch = (orgParams.inputs[0].batch().v == 8); // TODO: check why b16 not supported


        if (!bSupportedBatch)
        {
            return KernelsData();
        }

        return GetCommonKernelsData(params, optParams, DataLayout::fb, WeightsLayout::oi, FORCE_PRIORITY_6);
    }
}