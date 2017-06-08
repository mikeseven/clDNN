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

#include "pooling_kernel_ref.h"
 
namespace KernelSelector 
{
    ParamsKey PoolingKernelRef::GetSupportedKey() const
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
        k.SetPoolType(PoolType::MAX);
        k.SetPoolType(PoolType::AVG);
        k.SetPoolRemainder(PoolRemainder::FLOOR);
        k.SetPoolRemainder(PoolRemainder::CEIL);
        k.SetPoolKernelDividerMode(KernelDividerMode::FIXED);
        k.SetPoolKernelDividerMode(KernelDividerMode::DYNAMIC);
        return k;
    }

    KernelsData PoolingKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::POOLING);

        KernelData kd = KernelData::Default<PoolingParams>(params, 1);

        PoolingParams& newParams = *static_cast<PoolingParams*>(kd.params.get());
        const auto& pp = newParams.poolParams;
        
        const std::string kernel_id = params.layerID + std::to_string(UniqeID());
        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << "#define POOL_SIZE_X (" << pp.poolSize.x << ")\n"
            << "#define POOL_SIZE_Y (" << pp.poolSize.y << ")\n"
            << "#define POOL_PAD_X (" << pp.poolPad.x << ")\n"
            << "#define POOL_PAD_Y (" << pp.poolPad.y << ")\n"
            << "#define POOL_STRIDE_X (" << pp.poolStride.x << ")\n"
            << "#define POOL_STRIDE_Y (" << pp.poolStride.y << ")\n";

        jit << "#define " << toString(pp.poolType) << "_POOLING\n";
        jit << "#define " << toString(pp.divMode) << "_KERNEL_DIVIDER\n";

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x().v, out.y().v, out.feature().v*out.batch().v);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), kernel_id);
        kernel.args_desc = GetArgumentDesc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}