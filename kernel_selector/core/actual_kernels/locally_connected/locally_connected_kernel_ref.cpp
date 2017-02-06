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

#include "locally_connected_kernel_ref.h"
 
namespace KernelSelctor {

    ParamsKey LocallyConnectedKernelRef::GetSupportedKey() const
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

    KernelsData LocallyConnectedKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::LOCALLY_CONNECTED);

        KernelData kd = KernelData::Default<LocallyConnectedParams>(params, 1);

        LocallyConnectedParams& newParams = *static_cast<LocallyConnectedParams*>(kd.params.get());
        newParams.inputLayout = newParams.outputLayout = bfyx;

        std::stringstream jit;
        jit << GetBaseJit(newParams)
            << "#define KERNEL_WIDTH " << newParams.lcParams.filterSize.x << "\n"
            << "#define KERNEL_HEIGHT (" << newParams.lcParams.filterSize.y << ")\n"
            << "#define STRIDE_X (" << newParams.lcParams.stride.x << ")\n"
            << "#define STRIDE_Y (" << newParams.lcParams.stride.y << ")\n"
            << "#define INPUT_PADDING_X (" << newParams.lcParams.padding.x << ")\n"
            << "#define INPUT_PADDING_Y (" << newParams.lcParams.padding.y << ")\n";

        const auto& out = newParams.outDims;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x, out.y, out.z*out.w);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "locally_connected");
        kernel.args_desc = GetArgumentDesc(1, true, true);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}