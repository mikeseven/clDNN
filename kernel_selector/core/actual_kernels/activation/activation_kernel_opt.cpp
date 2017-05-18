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

#include "activation_kernel_opt.h"
 
namespace KernelSelctor {

    ParamsKey ActivationKernelOpt::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetSubGroupSupport();
        k.SetNumDims(3);
        return k;
    }

    KernelsData ActivationKernelOpt::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::ACTIVATION);

        KernelData kd = KernelData::Default<ActivationParams>(params, 1);

        ActivationParams& newParams = *static_cast<ActivationParams*>(kd.params.get());

        const uint32_t line_alignment = 4 / BytesPerElement(newParams.inputType);
        if ((newParams.activationFunc != ActivationFunction::NONE) ||
            (newParams.inDesc.pitches.x % line_alignment) != 0)
        {
            return{};
        }

        newParams.inputLayout = newParams.outputLayout = bfyx;

        static const int NUM_ROWS_WI = 1;
        static const int NUM_COLS_WI = 4;
        const uint nonWidthDim = newParams.inDims.Length() / newParams.inDims.x;

        std::stringstream jit;
        jit << GetBaseJit(newParams);

        jit << "#define NUM_ROWS_WI (" << NUM_ROWS_WI << ")\n"
            << "#define NUM_COLS_WI (" << NUM_COLS_WI << ")\n"
            << "#define INPUT_WIDTH (" << newParams.inDims.x << ")\n"
            << "#define INPUT_ROWS (" << nonWidthDim << ")\n"
            << "#define INPUT_ROWS_MOD_ROWS_WI " << nonWidthDim % NUM_ROWS_WI << "\n"
            << "#define INPUT_WIDTH_MOD_COLS_WI " << newParams.inDims.x % NUM_COLS_WI << "\n";

        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(
            (newParams.inDims.x + NUM_COLS_WI - 1) / NUM_COLS_WI,
            (nonWidthDim + NUM_ROWS_WI - 1) / NUM_ROWS_WI,
            newParams.outDims.w);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "activation");
        kernel.args_desc = GetArgumentDesc(1, false, false);

        kd.estimated_time = FORCE_PRIORITY_6;

        return{ kd };
    }
}