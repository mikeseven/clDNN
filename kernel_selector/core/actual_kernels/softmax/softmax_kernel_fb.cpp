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

#include "softmax_kernel_fb.h"
 
namespace KernelSelector 
{
    ParamsKey SoftmaxKernel_fb::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::yxfb);
        k.SetInputLayout(DataLayout::fb);
        k.SetOutputLayout(DataLayout::yxfb);
        k.SetOutputLayout(DataLayout::fb);
        k.SetSoftmaxDim(SoftmaxDim::X);         // in case that it can be flatten
        k.SetSoftmaxDim(SoftmaxDim::Y);
        k.SetSoftmaxDim(SoftmaxDim::FEATURE);
        k.SetBatchingSupport();
        return k;
    }

    IGKSoftmaxKernelBase::DispatchData SoftmaxKernel_fb::set_default(const SoftmaxParams& params, const OptionalParams& optParams) const
    {
        auto kd = IGKSoftmaxKernelBase::set_default(params, optParams);
        //start with 1 thread per data set
        kd.gws0 = kd.data_sets_count;
        kd.gws1 = 1;
        kd.items_num = kd.data_set_size;

        kd.norm_index = 1;

        // We have two units of data per work item in current implementation.
        auto local_mem_per_wi = 2 * (kd.fp16_unit_used ? sizeof(half_t) : sizeof(float));
        // Combining device execution and local memory restrictions to compute maximum possible LWS.
        auto max_lws = std::min(optParams.maxWorkGroupSize, optParams.maxLocalMemSize / local_mem_per_wi);

        kd.lws0 = kd.data_sets_count;
        // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory reads.
        while ((kd.items_num > 32 || kd.lws0 < kd.items_num) && (2 * kd.lws0 <= max_lws))
        {
            kd.lws0 *= 2;
            kd.items_num /= 2;
        }

        kd.gws0 = kd.lws0;
        kd.gws1 = 1;
        kd.leftovers = (kd.data_set_size * kd.data_sets_count) % kd.lws0;

        assert(kd.items_num > 0 && kd.lws0 && kd.gws0 > 0);

        return kd;
    }

    KernelsData SoftmaxKernel_fb::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {
        return GetCommonKernelsData(params, optParams, FORCE_PRIORITY_6);
    }
}