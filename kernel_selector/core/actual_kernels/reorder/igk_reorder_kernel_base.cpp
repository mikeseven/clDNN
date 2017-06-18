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

#include "kernel_selector_common.h"
#include "igk_reorder_kernel_base.h"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{
    inline uint32_t sub_group_size(WeightsLayout l)
    {
        switch (l)
        {
        case WeightsLayout::os_iyx_osv16:
        case WeightsLayout::os_i_osv16:
        case WeightsLayout::os_i_osv16__ai8:
        case WeightsLayout::i_yxs_os_yxsv2_osv16:
        case WeightsLayout::iy_xs_os_xsv2_osv16__ao32:
            return 16;
        case WeightsLayout::os_i_osv8__ai8:
        case WeightsLayout::iy_xs_os_xsv2_osv8__ao32:
            return 8;
        default:
            return 1;
        }
    }

    inline uint32_t sub_group_size(DataLayout l)
    {
        switch (l)
        {
        case DataLayout::bs_f_bsv16__af8:
            return 16;
        case DataLayout::bs_f_bsv8__af8:
            return 8;
        default:
            return 1;
        }
    }

    template<typename TensorT>
    static inline jit_constants _get_common_jit_constants(const TensorT& input, const TensorT& output, bool fp16_supported, uint32_t sub_group)
    {
        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("FP16_SUPPORTED",    fp16_supported),
            gpu::make_jit_constant("INPUT",             input),
            gpu::make_jit_constant("OUTPUT",            output),
            gpu::make_jit_constant("SUB_GROUP_SIZE",    sub_group),
        };

        return mem_consts;
    }

    jit_constants IGKReorderKernelBase::get_jit_constants(const ReorderWeightsParams& params) const
    {
        const auto& input = params.reorderParams.input;
        const auto& output = params.reorderParams.output;

        return _get_common_jit_constants(input, output, output.wtype == WeightsType::F16 || input.wtype == WeightsType::F16, sub_group_size(output.layout));
    }

    inline Datatype more_aqurate_data_type(Datatype a, Datatype b)
    {
        assert(a == Datatype::F16 || a == Datatype::F32);
        assert(b == Datatype::F16 || b == Datatype::F32);

        if (a == b)
        {
            return a;
        }

        return Datatype::F32;
    }

    jit_constants IGKReorderKernelBase::get_jit_constants(const ReorderParams& params) const
    {
        const auto& input = params.inputs[0];
        const auto& output = params.output;

        gpu::jit_constants mem_consts = _get_common_jit_constants(input, output, output.dtype == Datatype::F16 || input.dtype == Datatype::F16, sub_group_size(output.layout));
        mem_consts.add_constant(gpu::make_jit_constant("MEAN_SUBTRUCT_" + toString(params.reorderParams.mode), 1));

        Datatype calc_type = more_aqurate_data_type(input.dtype, output.dtype);
        if (params.reorderParams.mode == MeanSubtructMode::INSIDE_PARAMS)
        {
            mem_consts.add_constant(gpu::make_jit_constant("VALUE_TO_SUBTRACT", params.reorderParams.mean_values));
            calc_type = more_aqurate_data_type(calc_type, Datatype::F32);
        }
        else if (params.reorderParams.mode == MeanSubtructMode::IN_BUFFER)
        {
            mem_consts.add_constant(gpu::make_jit_constant("MEAN_SUBTRUCT", params.reorderParams.mean));
            calc_type = more_aqurate_data_type(calc_type, params.reorderParams.mean.dtype);
        }
        
        mem_consts.add_constants({
            gpu::make_jit_constant("CALC_TYPE",                      gpu::data_type_2_cl_type(calc_type)),
            gpu::make_jit_constant("TO_CALC_TYPE",      "convert_" + gpu::data_type_2_cl_type(calc_type)),
        });

        return mem_consts;
    }
}