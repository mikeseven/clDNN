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

#include "reorder_kernel.h"
#include "kernel_selector_utils.h"
 
namespace KernelSelector 
{
    ParamsKey ReorderKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetDifferentTypesSupport();
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    jit_constants ReorderKernelRef::GetJitConstants(const ReorderParams& params) const
    {
        auto jit = IGKReorderKernelBase::GetJitConstants(params);
        jit.merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));
        return jit;
    }

    KernelsData ReorderKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderParams>(params, 1);
        ReorderParams& newParams = *static_cast<ReorderParams*>(kd.params.get());

        std::string jit;

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID);

        try
        {
            auto cldnn_jit = GetJitConstants(newParams);
            jit = CreateJit(kernelName, cldnn_jit.get_definitions(), entry_point);
        }
        catch (const std::runtime_error&)
        {
            return KernelsData();
        }

        auto& kernel = kd.kernels[0];

        kernel.workGroups.global = GetTensorFriendlyWorkGroups(newParams.inputs[0]);
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);

        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, ROUND_ROBIN);
        kernel.argsDesc = GetArgsDesc(1, false, false);
        if (newParams.reorderParams.mode == MeanSubtructMode::IN_BUFFER)
        {
            kernel.argsDesc.data.push_back({ ArgumentDescpirtor::Types::BIAS, 0 });
        }

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}