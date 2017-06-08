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

#include "pooling_kernel_gpu_ref.h"
 
namespace KernelSelector 
{
    ParamsKey PoolingKernelGPURef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetInputLayout(DataLayout::yxfb);
        k.SetOutputLayout(DataLayout::bfyx);
        k.SetOutputLayout(DataLayout::yxfb);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        k.SetPoolType(PoolType::MAX);
        k.SetPoolType(PoolType::AVG);
        k.SetPoolRemainder(PoolRemainder::FLOOR);
        k.SetPoolRemainder(PoolRemainder::CEIL);
        k.SetPoolKernelDividerMode(KernelDividerMode::FIXED);
        k.SetPoolKernelDividerMode(KernelDividerMode::DYNAMIC);
        return k;
    }

    KernelsData PoolingKernelGPURef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::POOLING);

        const PoolingParams& orgParams = static_cast<const PoolingParams&>(params);

        const bool bSupportedActivation = orgParams.activationFunc == ActivationFunction::NONE;

        if (!bSupportedActivation)
        {
            return{};
        }
        
        DispatchData run_info;

        try
        {
            run_info = set_default(orgParams);
        }
        catch (const std::runtime_error&)
        {
            return{};
        }

        KernelData kd = KernelData::Default<PoolingParams>(params, 1);

        auto cldnn_jit = get_jit_constants(orgParams, run_info);
        auto entry_point = get_entry_point(kernel_name, orgParams.layerID);
        auto jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), entry_point);

        auto& kernel = kd.kernels[0];
        fill_cl_kernel_data(kernel, run_info, kernel_name, jit, entry_point);

        kd.estimated_time = FORCE_PRIORITY_9;

        return{ kd };
    }
}