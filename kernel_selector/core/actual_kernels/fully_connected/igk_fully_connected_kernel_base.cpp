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

#include "igk_fully_connected_kernel_base.h"
#include "kernel_selector_utils.h"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{
    jit_constants IGKFullyConnectedKernelBase::GetJitConstants(const FullyConnectedParams& params, const IGKFullyConnectedKernelBase::DispatchData& data) const
    {
        const auto& input = params.inputs[0];
        const auto x_size = input.Length() / input.batch().v;

        jit_constants mem_consts = GetCommonJitConstants(params);

        mem_consts.add_constants({
            gpu::make_jit_constant("INPUT_ELEMENTS_COUNT",      x_size),
            gpu::make_jit_constant("WEIGHTS_BATCH_NUM",         params.weights.ofm().v),
            gpu::make_jit_constant("BIAS_TERM",                 static_cast<int>(!params.bias.empty())),
            gpu::make_jit_constant("FILTER_X_PITCH",            params.weights.x().pitch),
            gpu::make_jit_constant("FILTER_Y_PITCH",            params.weights.y().pitch),
            gpu::make_jit_constant("FILTER_IFM_PITCH",          params.weights.ifm().pitch),
            gpu::make_jit_constant("FILTER_OFM_PITCH",          params.weights.ofm().pitch),
        });

        if (data.vload_kernel_type)
        {
            const auto batches_per_work_item = GetBatchesPerWorkItem(params);

            mem_consts.add_constant(gpu::make_jit_constant("NEURONS_PER_WORK_ITEM", GetNeuronsPerWorkItem(params))); // how many neurons for a single batch will a single work item produce
            mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", batches_per_work_item));             // how many batches will a single work item compute
            mem_consts.add_constant(gpu::make_jit_constant("OUTPUT_ELEMENTS_COUNT", params.output.Length() / params.output.batch().v));
        }

        return mem_consts;
    }

    IGKFullyConnectedKernelBase::DispatchData IGKFullyConnectedKernelBase::SetKernelData(const FullyConnectedParams& params) const
    {
        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].dtype == Datatype::F16;

        // Determine global work sizes.
        kd.gws0 = params.output.Length();
        kd.gws1 = kd.gws2 = 1;

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = kd.lws2 = 1;
        kd.vload_kernel_type = false;

        return kd;
    }

    KernelsData IGKFullyConnectedKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& optParams, DataLayout dl, WeightsLayout wl, float estimated_time) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        const auto& orgParams = static_cast<const FullyConnectedParams&>(params);
        const auto& orgOptParams = static_cast<const FullyConnectedOptionalParams&>(optParams);

        const bool bSupportedActivation = CheckActivationSupport(orgParams.activationFunc);
        
        bool bProperInput = orgParams.inputs[0].layout == dl;
        if (!bProperInput && orgParams.inputs[0].PaddingExists() == false)
        {
            bProperInput =
                (dl == DataLayout::fb && orgParams.inputs[0].layout == DataLayout::fyxb) ||
                (dl == DataLayout::bf && orgParams.inputs[0].layout == DataLayout::bfyx);
        }

        bool bProperWeights = orgParams.weights.layout == wl;
        if (!bProperWeights && orgParams.weights.PaddingExists() == false)
        {
            bProperWeights =
                (wl == WeightsLayout::io && orgParams.weights.layout == WeightsLayout::iyxo) ||
                (wl == WeightsLayout::oi && orgParams.weights.layout == WeightsLayout::oiyx);
        }

        const bool bSupportedLayout = orgOptParams.allowReorderInput || bProperInput;
        const bool bSupportedWeightsLayout = orgOptParams.allowWeightsReorder || bProperWeights;

        if (!bSupportedActivation || 
            !bSupportedLayout || 
            !bSupportedWeightsLayout)
        {
            return KernelsData();
        }

        KernelData kd = KernelData::Default<FullyConnectedParams>(params, 1);
        FullyConnectedParams& newParams = *static_cast<FullyConnectedParams*>(kd.params.get());

        if (!bProperInput)
        {
            newParams.inputs[0] = newParams.inputs[0].transform(dl);
            kd.reorderInput = true;
        }

        if (!bProperWeights)
        {
            newParams.weights = newParams.weights.transform(wl);
        }

        kd.kernels.resize(1);
        DispatchData run_info;
        std::string jit;

        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);

        try
        {
            run_info = SetDefault(newParams);
            auto cldnn_jit = GetJitConstants(newParams, run_info);
            jit = CreateJit(kernelName, cldnn_jit.get_definitions(), entry_point);
        }
        catch (const std::runtime_error&)
        {
            return KernelsData();
        }

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, run_info, kernelName, jit, entry_point, true, !orgParams.bias.empty());

        if (!bProperWeights)
        {
            bool succeed = SetWeightsReorderParams(orgParams, wl, kd.weightsReorderParams);

            if (!succeed)
            {
                return{};
            }
        }

        kd.estimatedTime = estimated_time;

        return{ kd };
    }
}