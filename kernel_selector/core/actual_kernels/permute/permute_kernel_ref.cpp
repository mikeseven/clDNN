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

#include "permute_kernel_ref.h"
#include "kernel_selector_utils.h" 
 
namespace KernelSelector 
{
    ParamsKey PermuteKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        //k.SetDifferentTypesSupport();
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    KernelsData PermuteKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<PermuteParams>(params, 1);
        PermuteParams& newParams = *static_cast<PermuteParams*>(kd.params.get());


        auto entry_point = GetEntryPoint(kernelName, newParams.layerID);
        auto cldnn_jit = MakePermuteJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        const auto& in = newParams.inputs[0];
        auto& kernel = kd.kernels[0];
        std::vector<size_t> gws;
        for (const auto& o : in.GetDims())
        {
            gws.push_back(o.v);
        }
        
        for (size_t i = gws.size(); i < 4; i++)
        {
            gws.push_back(1U);
        }

        kernel.workGroups.global = { gws[0], gws[1], gws[2] * gws[3] };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, ROUND_ROBIN);
        kernel.argsDesc = GetArgsDesc(1, false, false);
        
        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}