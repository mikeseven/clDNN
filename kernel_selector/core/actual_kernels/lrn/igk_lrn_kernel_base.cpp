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

#include "igk_lrn_kernel_base.h"
#include "api/CPP/tensor.hpp"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{
    jit_constants IGKLRNKernelBase::get_jit_constants(const LRNParams& params, IGKLRNKernelBase::DispatchData kd) const
    {
        jit_constants mem_consts = get_common_jit_constants(params, kd);

        const auto& np = params.lrnParams;

        // TODO: check why within channel behave different than across channel
        auto pad = (np.localSize - 1) / 2;
        //auto pad = (np.localSize) / 2;
        auto alpha = np.alpha;
        auto alpha_div_by_size = alpha / np.localSize;
        auto alpha_sign = std::signbit(alpha) ? -1.0f : 1.0f;
        // When used FP16 the value cannot be scaled afterwards by alpha (it must be scaled before computing sum of squares).
        auto alpha_abs_sqrt = std::sqrt(std::abs(alpha));
        auto alpha_div_by_size_abs_sqrt = std::sqrt(std::abs(alpha_div_by_size));

        mem_consts.add_constants({
            gpu::make_jit_constant("COUNT",                         params.inputs[0].Length()),
            gpu::make_jit_constant("P_SIZE",                        np.localSize),
            gpu::make_jit_constant("PAD",                           pad),
            gpu::make_jit_constant("ALPHA",                         kd.fp16_unit_used ? alpha_sign : alpha),
            gpu::make_jit_constant("ALPHA_DIV_BY_SIZE",             kd.fp16_unit_used ? alpha_sign : alpha_div_by_size),
            gpu::make_jit_constant("ALPHA_VAL_FACTOR",              kd.fp16_unit_used ? alpha_abs_sqrt : 1.0f),
            gpu::make_jit_constant("ALPHA_VAL_FACTOR_DIV_BY_SIZE",  kd.fp16_unit_used ? alpha_div_by_size_abs_sqrt : 1.0f),
            gpu::make_jit_constant("BETA",                          np.beta),
            gpu::make_jit_constant("K",                             np.k),
        });

        mem_consts.add_constant(gpu::make_jit_constant("LAYOUT_" + toString(params.inputs[0].layout), ""));

        return mem_consts;
    }

    IGKLRNKernelBase::DispatchData IGKLRNKernelBase::set_default(const LRNParams& params) const
    {
        const auto& output = params.output;

        DispatchData kd;

        kd.fp16_unit_used = params.inputs[0].dtype == Datatype::F16;
        // Determine global work sizes.
        kd.gws0 = output.batch().v * output.feature().v;    // B, F
        kd.gws1 = output.x().v;                             // X
        kd.gws2 = output.y().v;                             // Y
                                                            // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }
}