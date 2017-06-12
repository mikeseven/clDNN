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

#include "lrn_kernel_ref.h"
 
namespace KernelSelector 
{
    ParamsKey LRNKernelRef::GetSupportedKey() const
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
        k.SetLRNMode(LRNMode::WITHIN_CHANNEL);
        k.SetLRNMode(LRNMode::ACROSS_CHANNEL);
        k.SetLRNKernelDividerMode(KernelDividerMode::FIXED);
        return k;
    }

    KernelsData LRNKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::LRN);

        KernelData kd = KernelData::Default<LRNParams>(params, 1);

        LRNParams& newParams = *static_cast<LRNParams*>(kd.params.get());

        std::stringstream jit;
        
        const uint32_t round_norm_size = (newParams.lrnParams.localSize / 2) * 2 + 1;
        uint32_t numElement = round_norm_size * round_norm_size;

        if (newParams.lrnParams.normMode == LRNMode::ACROSS_CHANNEL)
        {
            jit << "#define ACROSS_MAPS\n";
            numElement = round_norm_size;
        }

        const float num_element_div = 1.f / numElement;
        const std::string kernel_id = params.layerID + std::to_string(UniqeID());

        jit << GetBaseJit(newParams, kernel_id)
            << "#define ROUND_NORM_SIZE (" << round_norm_size << ")\n"
            << "#define ROUND_NORM_HALF_SIZE (" << round_norm_size / 2 << ")\n"
            << "#define NUM_ELEMENTS_DIV (" << Float2Str(num_element_div) << ")\n"
            << "#define ALPHA (" << Float2Str(newParams.lrnParams.alpha) << ")\n"
            << "#define BETA (" << Float2Str(newParams.lrnParams.beta) << ")\n"
            << "#define NORM_K (1)\n";

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x().v, out.y().v, out.feature().v*out.batch().v);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), kernel_id, ROUND_ROBIN, "");
        kernel.args_desc = GetArgumentDesc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}