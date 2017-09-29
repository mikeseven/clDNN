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

#include "assign_patch_kernel_base.h"
#include "kernel_selector_utils.h" 

namespace KernelSelector
{
    bool AssignPatchKernelBase::Validate(const Params& p, const OptionalParams& o) const
    {
        if (p.GetType() != KernelType::ASSIGN_PATCH ||
            o.GetType() != KernelType::ASSIGN_PATCH)
        {
            return false;
        }

        const BaseParams& params = static_cast<const BaseParams&>(p);

        if (params.inputs.size() == 0)
        {
            return false;
        }

        return true;
    }

    JitConstants AssignPatchKernelBase::GetJitConstants(const AssignPatchParams& params) const
    {
        return MakeBaseParamsJitConstants(params);
    }

    KernelsData AssignPatchKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<AssignPatchParams>(params);
        AssignPatchParams& newParams = *static_cast<AssignPatchParams*>(kd.params.get());

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];

        const auto& source = newParams.inputs[0];
        const auto& out = newParams.output;

        kernel.workGroups.global = { source.Feature().v, out.X().v - source.X().v + 1, out.Y().v - source.Y().v + 1 };

        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, ROUND_ROBIN);
        kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}