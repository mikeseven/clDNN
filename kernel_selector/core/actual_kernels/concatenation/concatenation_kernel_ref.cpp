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

#include "concatenation_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{

    ParamsKey ConcatenationKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        k.SetConcatAxis(ConcatAxis::X);
        k.SetConcatAxis(ConcatAxis::Y);
        k.SetConcatAxis(ConcatAxis::FEATURE);
        k.SetConcatAxis(ConcatAxis::BATCH);
        return k;
    }

    KernelsData ConcatenationKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::CONCATENATION);

        const ConcatenationParams& orgParams = static_cast<const ConcatenationParams&>(params);

        const bool bSupportedActivation = orgParams.activationFunc == ActivationFunction::NONE;
        
        if (!bSupportedActivation)
        {
            return{};
        }

        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<ConcatenationParams>(params);

        auto cldnnJit = GetJitConstants(orgParams);
        auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entryPoint);

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}