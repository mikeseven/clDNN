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

#include "lrn_kernel_base.h"

namespace kernel_selector 
{
    bool LRNKernelBase::Validate(const Params& p, const OptionalParams& o) const
    {
        if (p.GetType() != KernelType::LRN ||
            o.GetType() != KernelType::LRN)
        {
            return false;
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MakeLRNJitConstants
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    inline JitConstants MakeLRNJitConstants(const lrn_params& params)
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        const auto& np = params.lrnParams;

        const auto padding = (np.localSize - 1) / 2;

        jit.AddConstants({
            MakeJitConstant("LOCAL_SIZE",   np.localSize),
            MakeJitConstant("PADDING",      padding),
            MakeJitConstant("ALPHA",        np.alpha),
            MakeJitConstant("BETA",         np.beta),
            MakeJitConstant("K",            np.k),
            MakeJitConstant(toString(np.divMode) + "_KERNEL_DIVIDER", ""),
            MakeJitConstant(toString(np.normMode), ""),
        });

        return jit;
    }

    JitConstants LRNKernelBase::GetJitConstants(const lrn_params& params, LRNKernelBase::DispatchData kd) const
    {
        JitConstants mem_consts = MakeBaseParamsJitConstants(params);

        const auto& np = params.lrnParams;

        const auto padding = (np.localSize - 1) / 2;

        mem_consts.AddConstants({
            MakeJitConstant("LOCAL_SIZE",   np.localSize),
            MakeJitConstant("PADDING",      padding),
            MakeJitConstant("ALPHA",        np.alpha),
            MakeJitConstant("BETA",         np.beta),
            MakeJitConstant("K",            np.k),
            MakeJitConstant(toString(np.divMode) + "_KERNEL_DIVIDER", ""),
            MakeJitConstant(toString(np.normMode), ""),
        });

        //auto pad = (np.localSize) / 2;
        auto alpha = np.alpha;
        auto alpha_div_by_size = alpha / np.localSize;
        auto alpha_sign = std::signbit(alpha) ? -1.0f : 1.0f;
        // When used FP16 the value cannot be scaled afterwards by alpha (it must be scaled before computing sum of squares).
        auto alpha_abs_sqrt = std::sqrt(std::abs(alpha));
        auto alpha_div_by_size_abs_sqrt = std::sqrt(std::abs(alpha_div_by_size));

        mem_consts.AddConstants({
            MakeJitConstant("ALPHA_AFTER_FACTORED",          kd.fp16UnitUsed ? alpha_sign : alpha),
            MakeJitConstant("ALPHA_DIV_BY_SIZE",             kd.fp16UnitUsed ? alpha_sign : alpha_div_by_size),
            MakeJitConstant("ALPHA_VAL_FACTOR",              kd.fp16UnitUsed ? alpha_abs_sqrt : 1.0f),
            MakeJitConstant("ALPHA_VAL_FACTOR_DIV_BY_SIZE",  kd.fp16UnitUsed ? alpha_div_by_size_abs_sqrt : 1.0f),
        });

        return mem_consts;
    }

    LRNKernelBase::DispatchData LRNKernelBase::SetDefault(const lrn_params& params) const
    {
        const auto& output = params.output;

        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;
        // Determine global work sizes.
        kd.gws0 = output.Batch().v * output.Feature().v;    // B, F
        kd.gws1 = output.X().v;                             // X
        kd.gws2 = output.Y().v;                             // Y
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

    KernelsData LRNKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options, float estimatedTime) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const lrn_params& orgParams = static_cast<const lrn_params&>(params);

        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<lrn_params>(params);

        auto cldnnJit = GetJitConstants(orgParams, runInfo);
        auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entryPoint);

        kd.estimatedTime = estimatedTime;

        return{ kd };
    }
}