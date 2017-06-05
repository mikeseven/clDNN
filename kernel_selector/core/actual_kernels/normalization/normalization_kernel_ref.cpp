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

#include "normalization_kernel_ref.h"
 
namespace KernelSelector 
{
    ParamsKey NormalizationKernelRef::GetSupportedKey() const
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
        k.SetNormalizationMode(NormalizationMode::WITHIN_CHANNEL);
        k.SetNormalizationMode(NormalizationMode::ACROSS_CHANNELS);
        return k;
    }

    KernelsData NormalizationKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::NORMALIZATION);

        KernelData kd = KernelData::Default<NormalizationParams>(params, 1);

        NormalizationParams& newParams = *static_cast<NormalizationParams*>(kd.params.get());

        std::stringstream jit;
        
        const uint32_t round_norm_size = (newParams.normParams.localSize / 2) * 2 + 1;
        uint32_t numElement = round_norm_size * round_norm_size;

        if (newParams.normParams.normMode == NormalizationMode::ACROSS_CHANNELS)
        {
            jit << "#define ACROSS_MAPS\n";
            numElement = round_norm_size;
        }

        const float num_element_div = 1.f / numElement;

        jit << GetBaseJit(newParams)
            << "#define ROUND_NORM_SIZE (" << round_norm_size << ")\n"
            << "#define ROUND_NORM_HALF_SIZE (" << round_norm_size / 2 << ")\n"
            << "#define NUM_ELEMENTS_DIV (" << Float2Str(num_element_div) << ")\n"
            << "#define ALPHA (" << Float2Str(newParams.normParams.alpha) << ")\n"
            << "#define BETA (" << Float2Str(newParams.normParams.beta) << ")\n"
            << "#define NORM_K (1)\n";

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x().v, out.y().v, out.feature().v*out.batch().v);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "normalization", "", ROUND_ROBIN, "");
        kernel.args_desc = GetArgumentDesc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}