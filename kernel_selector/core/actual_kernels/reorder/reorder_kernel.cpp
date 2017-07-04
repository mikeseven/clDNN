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

#include "reorder_kernel.h"
#include "kernel_selector_utils.h"
 
namespace KernelSelector 
{
    ParamsKey ReorderKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableDifferentTypes();
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    JitConstants ReorderKernelRef::GetJitConstants(const ReorderParams& params) const
    {
        auto jit = ReorderKernelBase::GetJitConstants(params);
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));
        return jit;
    }

    KernelsData ReorderKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderParams>(params);
        ReorderParams& newParams = *static_cast<ReorderParams*>(kd.params.get());

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);
        
        auto& kernel = kd.kernels[0];

        kernel.workGroups.global = GetTensorFriendlyWorkGroups(newParams.inputs[0]);
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);

        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, ROUND_ROBIN);
        kernel.argsDesc = GetArgsDesc(1, false, false);
        if (newParams.reorderParams.mode == MeanSubtructMode::IN_BUFFER)
        {
            kernel.argsDesc.data.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
        }

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}