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

#include "fully_connected_kernel_ref.h"
 
namespace KernelSelctor 
{
    ParamsKey FullyConnectedKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.SetInputLayout(bfyx);
        k.SetInputLayout(bx);
        k.SetOutputLayout(bx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetNumDims(4);
        return k;
    }

    KernelsData FullyConnectedKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        KernelData kd = KernelData::Default<FullyConnectedParams>(params, 1);

        FullyConnectedParams& newParams = *static_cast<FullyConnectedParams*>(kd.params.get());

        std::stringstream jit;
        jit << GetBaseJit(newParams)
            << GetFullyConnectedJit(newParams);

        const auto& out = newParams.outDims;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x, out.y);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "fc");
        kernel.args_desc = GetArgumentDesc(1, true, true);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}