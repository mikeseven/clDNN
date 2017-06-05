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

namespace KernelSelector {

    std::string CNNKernelBase::GetBaseJit(const BaseParams& params) const
    {
        std::stringstream jit;

        if (params.kernelID.empty())
        {
            jit << "#define KERNEL(name) __kernel void name\n"
                << "#define FUNC(name) name\n"
                << "#define FUNC_CALL(name) name\n";
        }
        else
        {
            jit << "#define KERNEL(name) __kernel void name##_" << params.kernelID << "\n"
                << "#define FUNC(name) name##_" << params.kernelID << "\n"
                << "#define FUNC_CALL(name) name##_" << params.kernelID << "\n";
        }
        

        jit << "#define ACTIVATION_FUNCTION_" << toString(params.activationFunc) << "\n"
            << "#define TYPE_" << toString(params.inputs[0].dtype) << "\n"
            << "#define NL_M (" << Float2Str(params.nlParams.m) << ")\n"
            << "#define NL_N (" << Float2Str(params.nlParams.n) << ")\n"
            << "#define INPUT_OFFSET (" << params.inputs[0].offset << ")\n"
            << "#define OUT_OFFSET (" << params.output.offset << ")\n";

        jit << "#define INPUT_WIDTH (" << params.inputs[0].x().v << ")\n"
            << "#define INPUT_HEIGHT (" << params.inputs[0].y().v << ")\n"
            << "#define INPUT_DEPTH (" << params.inputs[0].feature().v << ")\n"
            << "#define INPUT_BATCH (" << params.inputs[0].batch().v << ")\n"
            << "#define INPUT_Y_PITCH (" << params.inputs[0].y().pitch << ")\n"
            << "#define INPUT_FEATURE_PITCH (" << params.inputs[0].feature().pitch << ")\n"
            << "#define INPUT_BATCH_PITCH (" << params.inputs[0].batch().pitch << ")\n";

        jit << "#define OUT_WIDTH (" << params.output.x().v << ")\n"
            << "#define OUT_HEIGHT (" << params.output.y().v << ")\n"
            << "#define OUT_DEPTH (" << params.output.feature().v << ")\n"
            << "#define OUT_BATCH (" << params.output.batch().v << ")\n"
            << "#define OUT_Y_PITCH (" << params.output.y().pitch << ")\n"
            << "#define OUT_FEATURE_PITCH (" << params.output.feature().pitch << ")\n"
            << "#define OUT_BATCH_PITCH (" << params.output.batch().pitch << ")\n";

        return jit.str();
    }

    ArgumentDescpirtor CNNKernelBase::GetArgumentDesc(uint32_t num_of_input, bool use_weights, bool use_bias) const
    {
        ArgumentDescpirtor desc;

        for (uint32_t i = 0; i < num_of_input; i++)
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

    KernelString CNNKernelBase::GetKernelString(std::string name, std::string jit, std::string entry_point, std::string kernel_id, std::string exe_mode, std::string default_build_flags) const
    {
        KernelString kernel_string;

        auto codes = db.get(name);

        if (codes.size())
        {
            kernel_string.str = codes[0];
            kernel_string.jit = jit;
            kernel_string.options = exe_mode + " " + default_build_flags;
            kernel_string.entry_point = kernel_id.empty() ? entry_point : entry_point + "_" + kernel_id;
            kernel_string.batch_compilation = (kernel_id.empty() == false);
        }

        return kernel_string;
    }
}