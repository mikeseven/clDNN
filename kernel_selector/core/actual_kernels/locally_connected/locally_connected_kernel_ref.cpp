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
 
namespace KernelSelector {

    ParamsKey LocallyConnectedKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetOutputLayout(DataLayout::bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    KernelsData LocallyConnectedKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::LOCALLY_CONNECTED);

        KernelData kd = KernelData::Default<LocallyConnectedParams>(params, 1);

        LocallyConnectedParams& newParams = *static_cast<LocallyConnectedParams*>(kd.params.get());

        const std::string kernel_id = params.layerID + std::to_string(UniqeID());

        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << "#define KERNEL_WIDTH " << newParams.lcParams.filterSize.x << "\n"
            << "#define KERNEL_HEIGHT (" << newParams.lcParams.filterSize.y << ")\n"
            << "#define STRIDE_X (" << newParams.lcParams.stride.x << ")\n"
            << "#define STRIDE_Y (" << newParams.lcParams.stride.y << ")\n"
            << "#define INPUT_PADDING_X (" << newParams.lcParams.padding.x << ")\n"
            << "#define INPUT_PADDING_Y (" << newParams.lcParams.padding.y << ")\n";

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x().v, out.y().v, out.feature().v*out.batch().v);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), kernel_id);
        kernel.args_desc = GetArgumentDesc(1, true, true);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}