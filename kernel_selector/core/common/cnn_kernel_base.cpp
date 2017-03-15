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

#include "cnn_kernel_base.h"

namespace KernelSelctor {

    std::string CNNKernelBase::GetBaseJit(const BaseParams& params) const
    {
        std::stringstream jit;

        jit << "#define ACTIVATION_FUNCTION_" << toString(params.activationFunc) << "\n"
            << "#define TYPE_" << toString(params.inputType) << "\n"
            << "#define NL_M (" << std::to_string(params.nlParams.m) << ")\n"
            << "#define NL_N (" << std::to_string(params.nlParams.n) << ")\n"
            << "#define INPUT_OFFSET (" << params.inDesc.offset << ")\n"
            << "#define OUT_OFFSET (" << params.outDesc.offset << ")\n";

        if (params.inputLayout == bx)
        {
            jit << "#define INPUT_WIDTH (1)\n"
                << "#define INPUT_HEIGHT (1)\n"
                << "#define INPUT_DEPTH (" << params.inDims.x << ")\n"
                << "#define INPUT_BATCH (" << params.inDims.y << ")\n"
                << "#define INPUT_ROW_PITCH (1)\n"
                << "#define INPUT_SLICE_PITCH (1)\n"
                << "#define INPUT_BATCH_PITCH (" << params.inDesc.pitches.x << ")\n";
        }
        else
        {
            jit << "#define INPUT_WIDTH (" << params.inDims.x << ")\n"
                << "#define INPUT_HEIGHT (" << params.inDims.y << ")\n"
                << "#define INPUT_DEPTH (" << params.inDims.z << ")\n"
                << "#define INPUT_BATCH (" << params.inDims.w << ")\n"
                << "#define INPUT_ROW_PITCH (" << params.inDesc.pitches.x << ")\n"
                << "#define INPUT_SLICE_PITCH (" << params.inDesc.pitches.y << ")\n"
                << "#define INPUT_BATCH_PITCH (" << params.inDesc.pitches.z << ")\n";
        }

        if (params.outputLayout == bx)
        {
            jit << "#define OUT_WIDTH (1)\n"
                << "#define OUT_HEIGHT (1)\n"
                << "#define OUT_DEPTH (" << params.outDims.x << ")\n"
                << "#define OUT_BATCH (" << params.outDims.y << ")\n"
                << "#define OUT_ROW_PITCH (1)\n"
                << "#define OUT_SLICE_PITCH (1)\n"
                << "#define OUT_BATCH_PITCH (" << params.outDesc.pitches.x << ")\n";
        }
        else
        {
            jit << "#define OUT_WIDTH (" << params.outDims.x << ")\n"
                << "#define OUT_HEIGHT (" << params.outDims.y << ")\n"
                << "#define OUT_DEPTH (" << params.outDims.z << ")\n"
                << "#define OUT_BATCH (" << params.outDims.w << ")\n"
                << "#define OUT_ROW_PITCH (" << params.outDesc.pitches.x << ")\n"
                << "#define OUT_SLICE_PITCH (" << params.outDesc.pitches.y << ")\n"
                << "#define OUT_BATCH_PITCH (" << params.outDesc.pitches.z << ")\n";
        }

        return jit.str();
    }

    ArgumentDescpirtor CNNKernelBase::GetArgumentDesc(uint num_of_input, bool use_weights, bool use_bias) const
    {
        ArgumentDescpirtor desc;

        for (uint i = 0; i < num_of_input; i++)
        {
            desc.data.push_back({ ArgumentDescpirtor::Types::INPUT, 0 });
        }

        desc.data.push_back({ ArgumentDescpirtor::Types::OUTPUT, 0 });

        if (use_weights)
        {
            desc.data.push_back({ ArgumentDescpirtor::Types::WEIGHTS, 0 });
        }

        if (use_bias)
        {
            desc.data.push_back({ ArgumentDescpirtor::Types::BIAS, 0 });
        }

        return desc;
    }

    KernelString CNNKernelBase::GetKernelString(std::string name, std::string jit, std::string entry_point, std::string exe_mode) const
    {
        KernelString kernel_string;

        auto codes = db.get(name);

        if (codes.size())
        {
            kernel_string.str = codes[0];
            kernel_string.jit = jit;
            //kernel_string.options = " -cl-no-subgroup-ifp  -cl-unsafe-math-optimizations";
            kernel_string.options = exe_mode + " -cl-unsafe-math-optimizations";
            kernel_string.entry_point = entry_point;
        }

        return kernel_string;
    }
}