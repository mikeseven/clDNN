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

#include "softmax_kernel_ref.h"
 
namespace KernelSelctor 
{
    ParamsKey SoftmaxKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.SetInputLayout(bfyx);
        k.SetOutputLayout(bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetNumDims(4);
        return k;
    }

    KernelsData SoftmaxKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::SOFT_MAX);

        KernelData kd = KernelData::Default<SoftMaxParams>(params, 1);

        SoftMaxParams& newParams = *static_cast<SoftMaxParams*>(kd.params.get());
        newParams.inputLayout = newParams.outputLayout = bfyx;

        const auto& out = newParams.outDims;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x, out.y, out.z*out.w);
        kernel.kernel_string = GetKernelString(kernel_name, GetBaseJit(newParams), "softmax");
        kernel.args_desc = GetArgumentDesc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}