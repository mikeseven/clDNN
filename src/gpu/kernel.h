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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/CPP/profiling.hpp"
#include "api/CPP/primitive.hpp"

#include "memory_gpu.h"
#include "kernels_cache.h"
#include "event_impl.h"

#include "kernel_selector.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace KernelSelector;

namespace neural { namespace gpu {

class kernel : public context_holder 
{
    kernels_cache::kernel_id _kernel_id;

public:
    explicit kernel(std::shared_ptr<gpu_toolkit> context, const std::shared_ptr<KernelString>& kernel_string)
        : context_holder(context)
        , _kernel_id(
            context->get_kernels_cache().set_kernel_source(
                { kernel_string->jit,kernel_string->str }, 
                kernel_string->options, 
                kernel_string->entry_point, 
                kernel_string->batch_compilation)) 
    {}

    kernel(const kernel& other) : context_holder(other.context()), _kernel_id(other._kernel_id) {}

    kernel& operator=(const kernel& other) 
    {
        if (this == &other)
        {
            return *this;
        }

        _kernel_id = other._kernel_id;

        return *this;
    }

    struct kernel_arguments_desc
    {
        std::vector<const cldnn::memory*> inputs;
        const cldnn::memory* output = nullptr;
        const cldnn::memory* weights = nullptr;
        const cldnn::memory* bias = nullptr;
        const cldnn::memory* lookup_table = nullptr;
        const cldnn::memory* scale_table = nullptr;
        const cldnn::memory* slope = nullptr;
        uint32_t split = 0;
    };

    cldnn::refcounted_obj_ptr<cldnn::event_impl> run(
        const KernelSelector::clKernelData& kernel_data,
        const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& dependencies,
        const kernel_arguments_desc& args) const;
};

} }
