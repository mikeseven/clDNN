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

#include "activation_kernel_ref.h"
#include "kernel_selector_utils.h"
 
namespace KernelSelector {

    ParamsKey ActivationKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnablePRelu();
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    KernelsData ActivationKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::ACTIVATION);

        KernelData kd = KernelData::Default<ActivationParams>(params, 1);

        ActivationParams& newParams = *static_cast<ActivationParams*>(kd.params.get());
        const std::string kernel_id = params.layerID + std::to_string(UniqeID());

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { out.X().v, out.Y().v, out.Feature().v*out.Batch().v };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, GetBaseJit(newParams, kernel_id), kernel_id);
        kernel.argsDesc = GetArgumentDesc(1, false, false);
        if (newParams.activationFunc == ActivationFunction::PRELU)
        {
            kernel.argsDesc.data.push_back({ ArgumentDescpirtor::Types::SLOPE, 0 });
        }

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}