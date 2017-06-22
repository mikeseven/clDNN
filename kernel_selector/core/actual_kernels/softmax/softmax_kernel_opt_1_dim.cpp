﻿/*
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

#include "softmax_kernel_opt_1_dim.h"
 
namespace KernelSelector 
{
    ParamsKey SoftmaxKernelOpt1Dim::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::bf);
        k.SetOutputLayout(DataLayout::bf);
        k.SetSoftmaxDim(SoftmaxDim::FEATURE);
        k.SetOffsetSupport();
        return k;
    }

    KernelsData SoftmaxKernelOpt1Dim::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::SOFT_MAX);

        KernelData kd = KernelData::Default<SoftmaxParams>(params, 1);

        SoftmaxParams& newParams = *static_cast<SoftmaxParams*>(kd.params.get());

        const size_t maxLocalWorkGroup    = 32;
        const size_t dst_size             = newParams.output.Length();
        const size_t localWorkGroup       = std::min(std::max(dst_size, (size_t)1U), maxLocalWorkGroup);
        const size_t leftovers            = dst_size % localWorkGroup;
        const size_t globalWorkGroup      = dst_size - leftovers;
        const size_t itemsNum             = globalWorkGroup / localWorkGroup;
        const std::string kernel_id = params.layerID + std::to_string(UniqeID());

        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id);
        jit << "#define ITEMS_NUM (" << itemsNum << ")\n"
            << "#define LWS (" << localWorkGroup << ")\n"
            << "#define GWS (" << globalWorkGroup << ")\n"
            << "#define LEFTOVERS (" << leftovers << ")\n"
            ;

        if (newParams.inputs[0].GetDType() == Datatype::F16)
        {
            jit << "#define FP16_SUPPORTED (1)\n"
                << "#define FP16_UNIT_USED (1)\n";
        }
        else
        {
            jit << "#define FP16_SUPPORTED (0)\n"
                << "#define FP16_UNIT_USED (0)\n";
        }

        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = cl::NDRange(globalWorkGroup, 1, 1);
        kernel.workGroups.local = cl::NDRange(localWorkGroup, 1, 1);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.argsDesc = GetArgumentDesc(1, false, false);

        kd.estimatedTime = FORCE_PRIORITY_8;

        return{ kd };
    }
}