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

    KernelsData ReorderKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderParams>(params, 1);
        ReorderParams& newParams = *static_cast<ReorderParams*>(kd.params.get());

        std::string jit;

        auto entry_point = get_entry_point(kernel_name, newParams.layerID);

        try
        {
            auto cldnn_jit = get_jit_constants(newParams);
            jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), entry_point);
        }
        catch (const std::runtime_error&)
        {
            return KernelsData();
        }

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.batch().v, out.feature().v, out.x().v*out.y().v);
        kernel.kernel_string = get_kernel_string(kernel_name, jit, entry_point, ROUND_ROBIN);
        kernel.args_desc = get_args_desc(1, false, false);
        if (newParams.reorderParams.mode == MeanSubtructMode::IN_BUFFER)
        {
            kernel.args_desc.data.push_back({ ArgumentDescpirtor::Types::BIAS, 0 });
        }

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}