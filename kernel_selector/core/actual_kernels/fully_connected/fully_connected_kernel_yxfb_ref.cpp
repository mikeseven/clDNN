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

#include "fully_connected_kernel_yxfb_ref.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{
    ParamsKey FullyConnected_yxfb_ref::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputWeightsType(WeightsType::F16);
        k.SetInputWeightsType(WeightsType::F32);
        k.EnableAllInputLayout();
        k.SetOutputLayout(DataLayout::fb);
        k.SetBatchingSupport();
        k.SetBiasPerFeatureMap();
        k.SetNonBiasSupport();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        return k;
    }

    KernelsData FullyConnected_yxfb_ref::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        const auto& orgParams = static_cast<const FullyConnectedParams&>(params);
        const auto& orgOptParams = static_cast<const FullyConnectedOptionalParams&>(optParams);

        const bool bSupportedActivation = CheckActivationSupport(orgParams.activationFunc);
        const bool bProperInput = orgParams.inputs[0].layout == DataLayout::yxfb;
        const bool bSupportedLayout = orgOptParams.allowReorderInput || bProperInput;
        const bool bSupportedWeightsLayout = orgParams.weights.SimpleLayout();

        if (!bSupportedActivation || !bSupportedLayout || !bSupportedWeightsLayout)
        {
            return KernelsData();
        }

        KernelData kd = KernelData::Default<FullyConnectedParams>(params, 1);
        FullyConnectedParams& newParams = *static_cast<FullyConnectedParams*>(kd.params.get());
        
        if (!bProperInput)
        {
            newParams.inputs[0] = newParams.inputs[0].transform(DataLayout::yxfb);
            kd.reorderInput = true;
        }
        
        kd.kernels.resize(1);
        DispatchData run_info;
        std::string jit;
        
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);

        try
        {
            run_info = SetKernelData(newParams);
            auto cldnn_jit = GetJitConstants(newParams, run_info);
            cldnn_jit.AddConstant(MakeJitConstant("WEIGHTS_DIMS", newParams.weights.dims.size()));
            jit = CreateJit(kernelName, cldnn_jit, entry_point);
        }
        catch (const std::runtime_error& )
        {
            return KernelsData();
        }

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, run_info, kernelName, jit, entry_point, true, !orgParams.bias.empty());

        return{ kd };
    }
}