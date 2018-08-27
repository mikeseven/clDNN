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

#include "fully_connected_kernel_mmad_batched.h"
#include "kernel_selector_utils.h"
 
namespace kernel_selector 
{
    ParamsKey FullyConnected_mmad_batched::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableInputWeightsType(WeightsType::INT8);
        k.EnableInputLayout(DataLayout::byxf_af32);
        k.EnableOutputLayout(DataLayout::byxf_af32);
        k.EnableOutputLayout(DataLayout::bf);
        k.EnableBiasPerOutput();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnableInt8Quantization();
        k.EnableOutputCalibration();
        return k;
    }

    bool FullyConnected_mmad_batched::Validate(const Params& p, const optional_params& o) const
    {
        if (!FullyConnectedKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const fully_connected_params&>(p);

        size_t batch = params.inputs[0].Batch().v;
        if (batch != 64)
        {
            return false;
        }

        return true;
    }

    JitConstants FullyConnected_mmad_batched::GetJitConstants(const fully_connected_params& params, const DispatchData& runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws1));

        // pitch for special block format used in this kernel
        const size_t ifm_32_aligned = Align(params.weights.IFM().v, 32);
        const size_t filter_ofm_block_pitch = (ifm_32_aligned / 32) * params.weights.X().v * params.weights.Y().v * 4 * 8 * 8;
        jit.AddConstant(MakeJitConstant("FILTER_OFM_BLOCK_PITCH", filter_ofm_block_pitch));

        return jit;
    }

    std::unique_ptr<FullyConnected_mmad_batched::Parent::DispatchData> FullyConnected_mmad_batched::SetDefault(const fully_connected_params& params) const
    {
        auto runInfo = Parent::SetDefault(params);
        
        constexpr size_t sub_group_size = 8;

        const auto of_maps = params.output.Feature().v;
        const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);

        runInfo->gws0 = RoundUp(params.output.X().v * params.output.Y().v, 8) / 8;
        runInfo->gws1 = of_threads_per_batch * params.output.Batch().v;
        runInfo->gws2 = 1;

        runInfo->lws0 = 1;
        runInfo->lws1 = sub_group_size;
        runInfo->lws2 = 1;

        runInfo->effiency = FORCE_PRIORITY_1;
        return std::move(runInfo);
    }

    KernelsData FullyConnected_mmad_batched::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options, DataLayout::byxf_af32,
        { WeightsLayout::os_is_yx_isa8_osv8_isv4 }, FORCE_PRIORITY_1);
    }
}