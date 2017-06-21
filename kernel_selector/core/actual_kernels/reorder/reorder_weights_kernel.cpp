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

#include "kernel_selector_common.h"
#include "reorder_weights_kernel.h"
#include "kernel_selector_utils.h" 
 
namespace KernelSelector 
{
    ParamsKey ReorderWeightsKernel::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputWeightsType(WeightsType::F16);
        k.SetInputWeightsType(WeightsType::F32);
        k.SetOutputWeightsType(WeightsType::F16);
        k.SetOutputWeightsType(WeightsType::F32);
        //k.SetInputWeightsType(WeightsType::INT8);
        //k.SetOutputWeightsType(WeightsType::INT8);
        k.EnableAllWeightsLayout();
        k.SetDifferentTypesSupport();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        return k;
    }

    KernelsData ReorderWeightsKernel::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderWeightsParams>(params, 1);
        ReorderWeightsParams& newParams = *static_cast<ReorderWeightsParams*>(kd.params.get());

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

        const auto& out = newParams.reorderParams.output;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.ofm().v, out.ifm().v, out.x().v*out.y().v);
        kernel.work_groups.local = GetOptimalLocalWorkGroupSizes(kernel.work_groups.global);
        kernel.kernel_string = get_kernel_string(kernel_name, jit, entry_point, ROUND_ROBIN);
        kernel.args_desc = get_args_desc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}