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

#include "kernel_selector_common.h"
 
namespace KernelSelector {

    bool ArgumentDescpirtor::SetArguments(
        cl::Kernel& kernel,
        const SetArgumentParams& params) const
    {
        size_t inputIndex = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(data.size()); i++)
        {
            cl_int status = CL_INVALID_ARG_VALUE;

            switch (data[i].t)
            {
            case ArgumentDescpirtor::Types::INPUT:
                if (inputIndex < params.inputs.size() && params.inputs[inputIndex])
                {
                    status = kernel.setArg(i, *params.inputs[inputIndex]);
                    inputIndex++;
                }
                break;
            case ArgumentDescpirtor::Types::OUTPUT:
                if (params.output)
                {
                    status = kernel.setArg(i, *params.output);
                }
                break;
            case ArgumentDescpirtor::Types::WEIGHTS:
                if (params.weights)
                {
                    status = kernel.setArg(i, *params.weights);
                }
                break;
            case ArgumentDescpirtor::Types::BIAS:
                if (params.bias)
                {
                    status = kernel.setArg(i, *params.bias);
                }
                break;
            case ArgumentDescpirtor::Types::LOOKUP_TABLE:
                if (params.lookupTable)
                {
                    status = kernel.setArg(i, *params.lookupTable);
                }
                break;
            case ArgumentDescpirtor::Types::SCALE_TABLE:
                if (params.scaleTable)
                {
                    status = kernel.setArg(i, *params.scaleTable);
                }
                break;
            case ArgumentDescpirtor::Types::SLOPE:
                if (params.slope)
                {
                    status = kernel.setArg(i, *params.slope);
                }
                break;
            case ArgumentDescpirtor::Types::SPLIT:
                status = kernel.setArg(i, params.split);
                break;
            case ArgumentDescpirtor::Types::UINT8:
                status = kernel.setArg(i, data[i].v.u8);
                break;
            case ArgumentDescpirtor::Types::UINT16:
                status = kernel.setArg(i, data[i].v.u16);
                break;
            case ArgumentDescpirtor::Types::UINT32:
                status = kernel.setArg(i, data[i].v.u32);
                break;
            case ArgumentDescpirtor::Types::UINT64:
                status = kernel.setArg(i, data[i].v.u64);
                break;
            case ArgumentDescpirtor::Types::INT8:
                status = kernel.setArg(i, data[i].v.s8);
                break;
            case ArgumentDescpirtor::Types::INT16:
                status = kernel.setArg(i, data[i].v.s16);
                break;
            case ArgumentDescpirtor::Types::INT32:
                status = kernel.setArg(i, data[i].v.s32);
                break;
            case ArgumentDescpirtor::Types::INT64:
                status = kernel.setArg(i, data[i].v.s64);
                break;
            case ArgumentDescpirtor::Types::FLOAT32:
                status = kernel.setArg(i, data[i].v.f32);
                break;
            case ArgumentDescpirtor::Types::FLOAT64:
                status = kernel.setArg(i, data[i].v.f64);
                break;
            default:
                break;
            }

            if (status != CL_SUCCESS)
            {
                printf("Error set args\n");
                return false;
            }
        }

        return true;
    }

    binary_data clKernelData::GetBinary(context_device cl_context, program_cache& compiler) const
    {
        return compiler.get(cl_context, kernelString.jit + kernelString.str, kernelString.options);
    }

    std::string GetStringEnv(const char* varName)
    {
        std::string str;
#ifdef WIN32
        char* env = nullptr;
        size_t len = 0;
        errno_t err = _dupenv_s(&env, &len, varName);
        if (err == 0)
        {
            if (env != nullptr)
            {
                str = std::string(env);
            }
            free(env);
        }
#else
        const char *env = getenv(varName);
        if (env)
        {
            str = std::string(env);
        }
#endif

        return str;
    }
}