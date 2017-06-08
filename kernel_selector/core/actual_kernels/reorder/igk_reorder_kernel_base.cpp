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
    inline std::string weight_type_2_cl_type(WeightsType wType)
    {
        switch (wType)
        {
        case WeightsType::F16: return "half";
        case WeightsType::F32: return "float";
        case WeightsType::INT8: return "char";
        default: return "";
        }
    }

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

    jit_constants IGKReorderKernelBase::get_jit_constants(const ReorderWeightsParams& params) const
    {
        const auto& input = params.reorderParams.input;
        const auto& output = params.reorderParams.output;

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("SRC_TYPE",                  weight_type_2_cl_type(input.wtype)),
            gpu::make_jit_constant("DEST_TYPE",                 weight_type_2_cl_type(output.wtype)),
            gpu::make_jit_constant("FP16_SUPPORTED",            input.wtype == WeightsType::F16 || output.wtype == WeightsType::F16),
            gpu::make_jit_constant("INPUT_DIMS",                input.dims.size()),
            gpu::make_jit_constant("OUT_DIMS",                  output.dims.size()),
            gpu::make_jit_constant("INPUT_OFFSET",              input.offset),
            gpu::make_jit_constant("OUT_OFFSET",                output.offset),
            gpu::make_jit_constant("INPUT_X",                   input.x().v),
            gpu::make_jit_constant("INPUT_Y",                   input.y().v),
            gpu::make_jit_constant("OUTPUT_X",                  output.x().v),
            gpu::make_jit_constant("OUTPUT_Y",                  output.y().v),
            gpu::make_jit_constant("INPUT_X_PITCH",             input.x().pitch),
            gpu::make_jit_constant("INPUT_Y_PITCH",             input.y().pitch),
            gpu::make_jit_constant("INPUT_IFM_PITCH",           input.ifm().pitch),
            gpu::make_jit_constant("INPUT_OFM_PITCH",           input.ofm().pitch),
            gpu::make_jit_constant("OUT_X_PITCH",               output.x().pitch),
            gpu::make_jit_constant("OUT_Y_PITCH",               output.y().pitch),
            gpu::make_jit_constant("OUT_IFM_PITCH",             output.ifm().pitch),
            gpu::make_jit_constant("OUT_OFM_PITCH",             output.ofm().pitch),
            gpu::make_jit_constant("SIMPLE_INPUT",              input.SimpleLayout()),
            gpu::make_jit_constant("SIMPLE_OUTPUT",             output.SimpleLayout()),
            gpu::make_jit_constant("SUB_GROUP_SIZE",            sub_group_size(output.layout)),
        };

        mem_consts.add_constant(gpu::make_jit_constant("INPUT_LAYOUT_" + toString(input.layout), 1));
        mem_consts.add_constant(gpu::make_jit_constant("OUTPUT_LAYOUT_" + toString(output.layout), 1));

        return mem_consts;
    }
}