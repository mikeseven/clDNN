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

#include "igk_normalize_kernel_base.h"

namespace KernelSelector 
{
    jit_constants IGKNormalizeKernelBase::get_jit_constants(const NormalizeParams& params) const
    {
        gpu::jit_constants mem_consts = get_common_jit_constants(params);

        auto scale_feature_size = params.normParams.scale_table.feature().v;

        mem_consts.add_constants({
            gpu::make_jit_constant("SCALE_INDEX",   (scale_feature_size == 1) ? "0" : "f"),
            gpu::make_jit_constant("EPSILON",       params.normParams.epsilon),
        });

        return mem_consts;
    }

    IGKNormalizeKernelBase::DispatchData IGKNormalizeKernelBase::set_default(const NormalizeParams& params) const
    {
        const auto& output = params.output;

        DispatchData kd;

        kd.fp16_unit_used = params.inputs[0].dtype == Datatype::F16;

        if (params.normParams.normMode == NormalizeMode::WITHIN_SPATIAL)
        {
            kd.gws0 = output.x().v;
            kd.gws1 = output.y().v;
            kd.gws2 = output.batch().v;
        }
        else
        {
            kd.gws0 = output.batch().v;
            kd.gws1 = 1;
            kd.gws2 = 1;
        }

        kd.lws0 = kd.gws0;
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }

    KernelsData IGKNormalizeKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams&, float estimated_time) const
    {
        assert(params.GetType() == KernelType::NORMALIZE);

        const NormalizeParams& orgParams = static_cast<const NormalizeParams&>(params);

        if (!check_activation_support(orgParams.activationFunc))
        {
            return{};
        }

        DispatchData run_info;

        try
        {
            run_info = set_default(orgParams);
        }
        catch (const std::runtime_error&)
        {
            return{};
        }

        KernelData kd = KernelData::Default<NormalizeParams>(params, 1);

        auto cldnn_jit = get_jit_constants(orgParams);
        auto entry_point = get_entry_point(kernel_name, orgParams.layerID);
        auto jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), entry_point);

        auto& kernel = kd.kernels[0];
        fill_cl_kernel_data(kernel, run_info, kernel_name, jit, entry_point);
        kernel.args_desc.data.push_back({ ArgumentDescpirtor::Types::SCALE_TABLE, 0 });

        kd.estimated_time = estimated_time;

        return{ kd };
    }
}