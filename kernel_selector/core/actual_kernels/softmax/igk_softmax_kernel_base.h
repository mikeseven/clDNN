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

#pragma once

#include "igk_kernel_base.h"
#include "kernel_selector_params.h"

namespace KernelSelector 
{
    class IGKSoftmaxKernelBase : public IGKKernelBase
    {
    public:
        using IGKKernelBase::IGKKernelBase;
        virtual ~IGKSoftmaxKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {
            size_t items_num;
            size_t leftovers;
            size_t data_sets_count;
            size_t data_set_size;
            size_t norm_index; //which dimension (from in-memory representation) is normalized, e.g. for bfyx and softmax::normalize_f, it will be f's index == 2 (used only by naive kernel)
        };

        DispatchData _kernel_data;

    protected:
        jit_constants get_jit_constants(const SoftmaxParams& params, DispatchData kd) const;
        virtual DispatchData set_default(const SoftmaxParams& params, const OptionalParams& optParams) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams& optParams, float estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE) const;
    };
}