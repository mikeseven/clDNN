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
    class IGKConvolutionKernelBase : public IGKKernelBase
    {
    public:
        using IGKKernelBase::IGKKernelBase;
        virtual ~IGKConvolutionKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {
            size_t ofm_per_work_item; // how many output feature maps a single work item compute
            size_t batches_per_work_item; // how many batches will a single work item compute
            size_t block_width, block_height; // used for kernels processing blocks
            size_t prefetch;
            size_t input_block_array_size; ///< Number of elements in array of UNIT_TYPE that must be specified in kernel to store/cache input block.
            size_t input_block_width;      ///< Number of elements in X dimension stored/cached in input block.
            size_t leftovers;
        };
    
    protected:
        jit_constants get_jit_constants(const ConvolutionParams& params, DispatchData kd) const;
        DispatchData set_default(const ConvolutionParams& params) const;
        bool check_work_groups(const DispatchData&) const;
        bool check_pitch_for_split_only(const ConvolutionParams& params) const;
    };
}