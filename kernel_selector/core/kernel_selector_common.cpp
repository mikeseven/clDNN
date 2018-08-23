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

#include "kernel_selector_common.h"
#include <sstream>

namespace kernel_selector 
{
    std::string GetStringEnv(const char* varName)
    {
        std::string str;
#ifdef _WIN32
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

    std::string toString(NonLinearParams params)
    {
        std::stringstream s;
        s << "m" << params.m << "_n" << params.n;
        return s.str();
    }
    
    std::string toString(Tensor::Dim dim)
    {
        std::stringstream s;
        s << "v" << dim.v << "_p" << dim.pitch << "_" << dim.pad.before << "_" << dim.pad.after;
        return s.str();
    }

    std::string toString(DataTensor tensor)
    {
        std::stringstream s;
        s << toString(tensor.GetDType()) << "_";
        s << toString(tensor.GetLayout()) << "_";
        int i = 0;
        for (auto dim : tensor.GetDims())
        {
            s << "d" << i << "_" << toString(dim) << "_";
            i++;
        }
        return s.str();
    }
}