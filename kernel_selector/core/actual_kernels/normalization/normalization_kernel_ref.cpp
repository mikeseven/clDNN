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
 
namespace KernelSelctor 
{
    ParamsKey NormalizationKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.SetInputLayout(bfyx);
        k.SetOutputLayout(bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetNumDims(4);
        k.SetNormalizationMode(NormalizationMode::WITHIN_CHANNEL);
        k.SetNormalizationMode(NormalizationMode::ACROSS_CHANNELS);
        return k;
    }

    KernelsData NormalizationKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::NORMALIZATION);

        KernelData kd = KernelData::Default<NormalizationParams>(params, 1);

        NormalizationParams& newParams = *static_cast<NormalizationParams*>(kd.params.get());
        newParams.inputLayout = newParams.outputLayout = bfyx;

        std::stringstream jit;
        
        const uint round_norm_size = (newParams.normParams.localSize / 2) * 2 + 1;
        uint numElement = round_norm_size * round_norm_size;

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

        const auto& out = newParams.outDims;
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(out.x, out.y, out.z*out.w);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "normalization", ROUND_ROBIN, "");
        kernel.args_desc = GetArgumentDesc(1, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}