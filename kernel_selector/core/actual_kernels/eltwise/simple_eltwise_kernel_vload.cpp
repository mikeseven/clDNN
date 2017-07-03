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

#include "simple_eltwise_kernel_vload.h"
#include "kernel_selector_utils.h" 

namespace KernelSelector {

    ParamsKey SimpleEltwiseKernel_vload::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableBatching();
        return k;
    }

    JitConstants SimpleEltwiseKernel_vload::get_jit_constants(const EltwiseParams& params) const
    {
        auto jit = MakeEltwiseJitConstants(params);
        jit.AddConstant(MakeJitConstant(toString(params.eltwiseParams.operations[0].mode) + "_MODE_USED", 1));
        return jit;
    }

    bool SimpleEltwiseKernel_vload::Validate(const Params& params, const OptionalParams&) const
    {
        if (params.GetType() != KernelType::ELTWISE)
        {
            return false;
        }

        const EltwiseParams& ewParams = static_cast<const EltwiseParams&>(params);

        if (!CheckActivationSupport(ewParams.activationFunc) ||
            !CheckInputsOutputNoPitchSameDims(ewParams) ||
            ewParams.eltwiseParams.operations.size() != 1 ||
            ewParams.inputs.size() != 2)
        {
            return false;
        }

        switch (ewParams.eltwiseParams.operations[0].mode)
        {
        case EltwiseMode::ADD:
        case EltwiseMode::SUB:
        case EltwiseMode::MUL:
        case EltwiseMode::MAX:
            break;
        default:
            return false;
        }

        return true;
    }

    KernelsData SimpleEltwiseKernel_vload::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {
        if (!Validate(params, optParams))
        {
            return{};
        }

        KernelData kd = KernelData::Default<EltwiseParams>(params);
        EltwiseParams& newParams = *static_cast<EltwiseParams*>(kd.params.get());

        std::string jit;

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID);

        try
        {
            auto cldnn_jit = get_jit_constants(newParams);
            jit = CreateJit(kernelName, cldnn_jit, entry_point);
        }
        catch (const std::runtime_error&)
        {
            return KernelsData();
        }

        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { std::max(newParams.inputs[0].LogicalSize()/8, (size_t)1), 1, 1 };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, ROUND_ROBIN);
        kernel.argsDesc = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

        kd.estimatedTime = FORCE_PRIORITY_8;

        return{ kd };
    }
}