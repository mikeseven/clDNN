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

#include "convert_kernel_ref.h"
 
namespace KernelSelector {

    ParamsKey ConvertKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    KernelsData ConvertKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::CONVERT);

        KernelData kd = KernelData::Default<ConvertParams>(params, 1);

        ConvertParams& newParams = *static_cast<ConvertParams*>(kd.params.get());
        const std::string kernel_id = params.layerID + std::to_string(UniqeID());

        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << "#define CONVERT_TYPE_" << toString(newParams.convertParams.covertType) << "\n";

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x().v, out.y().v, out.feature().v*out.batch().v);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), kernel_id);
        kernel.args_desc = GetArgumentDesc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}