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

#include "reshape_kernel_ref.h"
 
namespace KernelSelector 
{
    ParamsKey ReshapeKernelRef::GetSupportedKey() const
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

    KernelsData ReshapeKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderBaseParams>(params, 1);
        ReorderBaseParams& newParams = *static_cast<ReorderBaseParams*>(kd.params.get());

        std::string jit;

        auto entry_point = get_entry_point(kernel_name, newParams.layerID);

        try
        {
            auto cldnn_jit = get_common_jit_constants(newParams);
            jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), entry_point);
        }
        catch (const std::runtime_error&)
        {
            return KernelsData();
        }

        const auto& in = newParams.inputs[0];
        auto& kernel = kd.kernels[0];
        std::vector<size_t> gws;
        for (const auto& o : in.dims)
        {
            gws.push_back(o.v);
        }
        
        for (size_t i = gws.size(); i < 4; i++)
        {
            gws.push_back(1U);
        }

        kernel.work_groups.global = cl::NDRange(gws[0], gws[1], gws[2]*gws[3]);
        kernel.kernel_string = get_kernel_string(kernel_name, jit, entry_point, ROUND_ROBIN);
        kernel.args_desc = get_args_desc(1, false, false);
        
        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}