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

#include "deconvolution_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{

    ParamsKey DeconvolutionKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetInputWeightsType(WeightsType::F16);
        k.SetInputWeightsType(WeightsType::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::yxfb);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetOutputLayout(DataLayout::yxfb);
        k.SetOutputLayout(DataLayout::bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBiasPerFeatureMap();
        //k.SetBiasPerOutput();
        k.SetNonBiasSupport();
        k.SetBatchingSupport();
        k.SetSplitSupport();
        //k.SetDilationSupport();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        return k;
    }

    KernelsData DeconvolutionKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::DECONVOLUTION);

        const DeconvolutionParams& orgParams = static_cast<const DeconvolutionParams&>(params);

        const bool bSupportedActivation = CheckActivationSupport(orgParams.activationFunc);

        const bool bSupportedWeightsLayout =
            orgParams.weights.layout == WeightsLayout::yxio ||
            orgParams.weights.layout == WeightsLayout::iyxo ||
            orgParams.weights.layout == WeightsLayout::oyxi ||
            orgParams.weights.layout == WeightsLayout::oiyx;
        
        if (!bSupportedActivation || !bSupportedWeightsLayout)
        {
            return{};
        }

        DispatchData run_info = SetDefault(orgParams);
        KernelData kd = KernelData::Default<DeconvolutionParams>(params, 1);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit.get_definitions(), entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, run_info, kernelName, jit, entry_point, true, !orgParams.bias.empty());
        kernel.argsDesc.data.push_back({ ArgumentDescpirtor::Types::SPLIT, 0 });

        kd.estimatedTime = run_info.effiency;

        return{ kd };
    }
}