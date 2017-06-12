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

#pragma once

#include "kernel_base.h"
#include <sstream>
#include <assert.h>

namespace KernelSelector {

    class CNNKernelBase : public KernelBase
    {
    public:
        using KernelBase::KernelBase;
        virtual ~CNNKernelBase() {}

    protected:
        std::string GetBaseJit(const BaseParams& params, const std::string& kernel_id) const;
        ArgumentDescpirtor GetArgumentDesc(uint32_t num_of_input, bool use_weights, bool use_bias) const;
        KernelString GetKernelString(std::string kernel_name, std::string jit, std::string entry_point, std::string exe_mode = ROUND_ROBIN, std::string default_build_flags = "-cl-unsafe-math-optimizations") const;
        static std::string Float2Str(const float f)
        {
            return std::to_string(f) + "f";
        }
    };
}