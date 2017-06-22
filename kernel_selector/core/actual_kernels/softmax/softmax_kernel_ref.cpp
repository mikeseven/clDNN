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

#include "softmax_kernel_ref.h"
#include "kernel_selector_utils.h" 
 
namespace KernelSelector 
{
    ParamsKey SoftmaxKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetInputLayout(DataLayout::yxfb);
        k.SetInputLayout(DataLayout::bf);
        k.SetInputLayout(DataLayout::fb);
        k.SetOutputLayout(DataLayout::bfyx);
        k.SetOutputLayout(DataLayout::yxfb);
        k.SetOutputLayout(DataLayout::bf);
        k.SetOutputLayout(DataLayout::fb);
        k.SetSoftmaxDim(SoftmaxDim::X);
        k.SetSoftmaxDim(SoftmaxDim::Y);
        k.SetSoftmaxDim(SoftmaxDim::FEATURE);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    KernelsData SoftmaxKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::SOFT_MAX);

        KernelData kd = KernelData::Default<SoftmaxParams>(params, 1);

        SoftmaxParams& newParams = *static_cast<SoftmaxParams*>(kd.params.get());
        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        const std::string kernel_id = params.layerID + std::to_string(UniqeID());
        auto jit = GetBaseJit(newParams, kernel_id);
        switch (newParams.smParams.dim)
        {
        case SoftmaxDim::X:
            jit +=
                "#define INPUT_OTHER0_PITCH     INPUT_Y_PITCH\n"
                "#define INPUT_OTHER1_PITCH     INPUT_FEATURE_PITCH\n"
                "#define INPUT_CLASS_PITCH      INPUT_X_PITCH\n"
                "#define INPUT_CLASS_NUM        INPUT_SIZE_X\n"
                "#define OUTPUT_OTHER0_PITCH    OUTPUT_Y_PITCH\n"
                "#define OUTPUT_OTHER1_PITCH    OUTPUT_FEATURE_PITCH\n"
                "#define OUTPUT_CLASS_PITCH     OUTPUT_X_PITCH\n";
            kernel.workGroups.global = cl::NDRange(out.y().v, out.feature().v, out.batch().v);
            break;
        case SoftmaxDim::Y:
            jit +=
                "#define INPUT_OTHER0_PITCH     INPUT_X_PITCH\n"
                "#define INPUT_OTHER1_PITCH     INPUT_FEATURE_PITCH\n"
                "#define INPUT_CLASS_PITCH      INPUT_Y_PITCH\n"
                "#define INPUT_CLASS_NUM        INPUT_SIZE_Y\n"
                "#define OUTPUT_OTHER0_PITCH    OUTPUT_X_PITCH\n"
                "#define OUTPUT_OTHER1_PITCH    OUTPUT_FEATURE_PITCH\n"
                "#define OUTPUT_CLASS_PITCH     OUTPUT_Y_PITCH\n";
            kernel.workGroups.global = cl::NDRange(out.x().v, out.feature().v, out.batch().v);
            break;
        case SoftmaxDim::FEATURE:
            jit +=
                "#define INPUT_OTHER0_PITCH     INPUT_X_PITCH\n"
                "#define INPUT_OTHER1_PITCH     INPUT_Y_PITCH\n"
                "#define INPUT_CLASS_PITCH      INPUT_FEATURE_PITCH\n"
                "#define INPUT_CLASS_NUM        INPUT_FEATURE_NUM\n"
                "#define OUTPUT_OTHER0_PITCH    OUTPUT_X_PITCH\n"
                "#define OUTPUT_OTHER1_PITCH    OUTPUT_Y_PITCH\n"
                "#define OUTPUT_CLASS_PITCH     OUTPUT_FEATURE_PITCH\n";
            kernel.workGroups.global = cl::NDRange(out.x().v, out.y().v, out.batch().v);
            break;
        default:
            break;
        }

        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit, kernel_id);
        kernel.argsDesc = GetArgumentDesc(1, false, false);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}