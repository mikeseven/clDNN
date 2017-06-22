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

#include "igk_softmax_kernel_base.h"

namespace KernelSelector 
{
    JitConstants IGKSoftmaxKernelBase::GetJitConstants(const SoftmaxParams& params, IGKSoftmaxKernelBase::DispatchData kd) const
    {
        JitConstants mem_consts = MakeSoftmaxJitConstants(params);

        mem_consts.AddConstants({
            MakeJitConstant("ITEMS_NUM",      kd.itemsNum),
            MakeJitConstant("LWS",            kd.lws0),
            MakeJitConstant("GWS",            kd.gws0),
            MakeJitConstant("DATA_SETS_COUNT",kd.dataSetsCount),
            MakeJitConstant("DATA_SET_SIZE",  kd.dataSetSize),
            MakeJitConstant("LEFTOVERS",      kd.leftovers),
        });

        return mem_consts;
    }

    static bool validate(const SoftmaxParams& params)
    {
        const auto& input = params.inputs[0];

        if (params.activationFunc != ActivationFunction::NONE)
        {
            return false;
        }
        
        if (input.layout == DataLayout::bf ||
            input.layout == DataLayout::fb)
        {
            return true;
        }

        switch (params.smParams.dim)
        {
        case SoftmaxDim::X:         return input.y().v == 1 && input.feature().v == 1;
        case SoftmaxDim::Y:         return input.x().v == 1 && input.feature().v == 1;
        case SoftmaxDim::FEATURE:   return input.x().v == 1 && input.y().v == 1;
        default:                    return false;
        }
    }

    IGKSoftmaxKernelBase::DispatchData IGKSoftmaxKernelBase::SetDefault(const SoftmaxParams& params, const OptionalParams&) const
    {
        const auto& input = params.inputs[0];

        DispatchData kd;
        
        kd.gws0 = 1;
        kd.gws1 = 1;
        kd.gws2 = 1;

        kd.lws0 = 1;
        kd.lws1 = 1;
        kd.lws2 = 1;


        kd.fp16UnitUsed = input.dtype == Datatype::F16;
        kd.leftovers = 0;
        kd.itemsNum = 0;

        kd.dataSetsCount = 0;

        // currently all derived kernels support bf/fb only
        auto flatten_input = input.flatten_fyx_2_f();
        kd.dataSetSize = flatten_input.feature().v;
        kd.dataSetsCount = input.batch().v;

        return kd;
    }

    KernelsData IGKSoftmaxKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& optParams, float estimated_time) const
    {
        assert(params.GetType() == KernelType::SOFT_MAX);

        const SoftmaxParams& orgParams = static_cast<const SoftmaxParams&>(params);

        if (!validate(orgParams))
        {
            return{};
        }

        DispatchData run_info;

        try
        {
            run_info = SetDefault(orgParams, optParams);
        }
        catch (const std::runtime_error&)
        {
            return{};
        }

        KernelData kd = KernelData::Default<SoftmaxParams>(params, 1);

        auto cldnn_jit = GetJitConstants(orgParams, run_info);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, run_info, kernelName, jit, entry_point);

        kd.estimatedTime = estimated_time;

        return{ kd };
    }
}