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

#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
 
namespace KernelSelector 
{
    class KernelBase
    {
    public:
        KernelBase(const std::string name) : kernel_name(name) {}
        virtual ~KernelBase() {}

        virtual KernelsData GetKernelsData(const Params& params, const OptionalParams& options) const = 0;
        virtual ParamsKey GetSupportedKey() const = 0;
        virtual const std::string GetName() const { return kernel_name; }
    
    protected:
        static const primitive_db db;
        const std::string kernel_name;

        static size_t UniqeID() { return counter++; } // TODO: use interlocked
        
    private:
        static size_t counter;
    };
}