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
 
namespace KernelSelector 
{
#if 0
    ParamsKey ReorderKernel::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    KernelsData ReorderKernel::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderVxParams>(params, 1);

        ReorderVxParams& newParams = *static_cast<ReorderVxParams*>(kd.params.get());

        std::stringstream jit;
        jit << GetBaseJit(newParams);
        jit << "#define REORDER_MODE_" << toString(newParams.reorderParams.mode);

        const auto& in = newParams.input;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(in.x().v, in.y().v, in.feature().v*in.batch().v);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "reorder");
        kernel.args_desc = GetArgumentDesc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
#endif
}