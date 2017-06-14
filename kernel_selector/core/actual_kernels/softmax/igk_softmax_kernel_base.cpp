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
#include "api/CPP/tensor.hpp"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{
    jit_constants IGKSoftmaxKernelBase::get_jit_constants(const SoftmaxParams& params, IGKSoftmaxKernelBase::DispatchData kd) const
    {
        jit_constants mem_consts = get_common_jit_constants(params, kd);

        mem_consts.add_constants({
            gpu::make_jit_constant("ITEMS_NUM",      kd.items_num),
            gpu::make_jit_constant("LWS",            kd.lws0),
            gpu::make_jit_constant("GWS",            kd.gws0),
            gpu::make_jit_constant("DATA_SETS_COUNT",kd.data_sets_count),
            gpu::make_jit_constant("DATA_SET_SIZE",  kd.data_set_size),
            gpu::make_jit_constant("LEFTOVERS",      kd.leftovers),
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

    IGKSoftmaxKernelBase::DispatchData IGKSoftmaxKernelBase::set_default(const SoftmaxParams& params, const OptionalParams&) const
    {
        const auto& input = params.inputs[0];

        DispatchData kd;
        
        kd.gws0 = 1;
        kd.gws1 = 1;
        kd.gws2 = 1;

        kd.lws0 = 1;
        kd.lws1 = 1;
        kd.lws2 = 1;


        kd.fp16_unit_used = input.dtype == Datatype::F16;
        kd.leftovers = 0;
        kd.items_num = 0;

        kd.data_sets_count = 0;

        // currently all derived kernels support bf/fb only
        auto flatten_input = input.flatten_fyx_2_f();
        kd.data_set_size = flatten_input.feature().v;
        kd.data_sets_count = input.batch().v;

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
            run_info = set_default(orgParams, optParams);
        }
        catch (const std::runtime_error&)
        {
            return{};
        }

        KernelData kd = KernelData::Default<SoftmaxParams>(params, 1);

        auto cldnn_jit = get_jit_constants(orgParams, run_info);
        auto entry_point = get_entry_point(kernel_name, orgParams.layerID);
        auto jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), entry_point);

        auto& kernel = kd.kernels[0];
        fill_cl_kernel_data(kernel, run_info, kernel_name, jit, entry_point);

        kd.estimated_time = estimated_time;

        return{ kd };
    }
}