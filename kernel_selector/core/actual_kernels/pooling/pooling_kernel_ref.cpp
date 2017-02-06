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
 
namespace KernelSelctor 
{
    ParamsKey PoolingKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.SetInputLayout(bfyx);
        k.SetOutputLayout(bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetNumDims(4);
        k.SetPoolType(PoolType::MAX);
        k.SetPoolType(PoolType::AVG);
        k.SetPoolRemainder(PoolRemainder::FLOOR);
        k.SetPoolRemainder(PoolRemainder::CEIL);
        return k;
    }

    KernelsData PoolingKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::POOLING);

        KernelData kd = KernelData::Default<PoolingParams>(params, 1);

        PoolingParams& newParams = *static_cast<PoolingParams*>(kd.params.get());
        newParams.inputLayout = newParams.outputLayout = bfyx;

        std::stringstream jit;
        jit << GetBaseJit(newParams)
            << "#define POOL_SIZE_X (" << newParams.poolParams.poolSize.x << ")\n"
            << "#define POOL_SIZE_Y (" << newParams.poolParams.poolSize.y << ")\n"
            << "#define POOL_PAD_X (" << newParams.poolParams.poolPad.x << ")\n"
            << "#define POOL_PAD_Y (" << newParams.poolParams.poolPad.y << ")\n"
            << "#define POOL_STRIDE_X (" << newParams.poolParams.poolStride.x << ")\n"
            << "#define POOL_STRIDE_Y (" << newParams.poolParams.poolStride.y << ")\n";

        if (newParams.poolParams.poolType == PoolType::MAX)
        {
            jit << "#define MAX_POOLING\n";
        }

        const auto& out = newParams.outDims;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x, out.y, out.z*out.w);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "pooling");
        kernel.args_desc = GetArgumentDesc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}