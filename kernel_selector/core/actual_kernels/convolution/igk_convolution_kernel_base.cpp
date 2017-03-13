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

#include "igk_convolution_kernel_base.h"
#include "api/tensor.hpp"

namespace KernelSelctor 
{
    void CPUIGKConvolutionReorder::Execute(void* input, std::size_t, void* output, std::size_t) const
    {
        assert(input_layout == WeightsReorderLayout::oiyx);
        assert(output_layout == WeightsReorderLayout::os_iyx_osv16);
        constexpr uint32_t sub_group_size = 16;

        short* input_ptr = (short*)input;
        short* output_ptr = (short*)output;
        const auto bpp = BytesPerElement(params->inputType);
        const uint32_t of_threads_per_batch = round_up_to(params->outDims.z, sub_group_size);

        uint size[5] = {
            params->outDims.w,
            params->outDims.z,
            params->inDims.z,
            params->convParams.filterSize.x,
            params->convParams.filterSize.y,
        };
        for (uint32_t ofm = 0; ofm < params->outDims.z; ofm++)
        {
            for (uint32_t ifm = 0; ifm < params->inDims.z; ifm++)
            {
                for (uint32_t y = 0; y < params->convParams.filterSize.y; y++)
                {
                    for (uint32_t x = 0; x < params->convParams.filterSize.x; x++)
                    {
                        std::size_t input_idx = x + size[3] * (y + size[4] * (ifm + size[2] * ofm));
                        const uint slice_id = ofm / sub_group_size;
                        const uint id_in_slice = ofm % sub_group_size;
                        std::size_t output_idx = id_in_slice + 16 * (x + size[3] * (y + size[4] * (ifm + slice_id * size[2])));
                        //assert(input_idx*bpp < input_size && output_idx*bpp < output_size);
                        output_ptr[output_idx] = input_ptr[input_idx];
                    }
                }
            }
        }
    }

    std::size_t CPUIGKConvolutionReorder::GetNewWeightBufferSizeInBytes() const
    {
        constexpr uint32_t sub_group_size = 16;
        return
            params->convParams.filterSize.x *
            params->convParams.filterSize.y *
            params->inDims.z *
            round_up_to(params->outDims.z, sub_group_size) *
            BytesPerElement(params->inputType);
    }

    cldnn::tensor desc_2_tensor(const TensorDesc& desc, cldnn::format f)
    {
        auto stride_x = desc.pitches.x;
        auto stride_y = desc.pitches.y / desc.pitches.x;
        auto stride_z = desc.pitches.z / desc.pitches.y;
        auto stride_w = desc.pitches.w / desc.pitches.z;

        return{ f, 
            {
                (cldnn::tensor::value_type)stride_w,
                (cldnn::tensor::value_type)stride_z,
                (cldnn::tensor::value_type)stride_y,
                (cldnn::tensor::value_type)stride_x
            } 
        };
    }

    cldnn::tensor dim_2_tensor(const uDims& dim, cldnn::format f)
    {
        assert(f == cldnn::format::bfyx);
        return{ f,
        {
            (cldnn::tensor::value_type)dim.w,
            (cldnn::tensor::value_type)dim.z,
            (cldnn::tensor::value_type)dim.y,
            (cldnn::tensor::value_type)dim.x
        }
        };
    }

    cldnn::format params_2_cldnn(DataLayout l)
    {
        switch (l)
        {
        case KernelSelctor::x: return cldnn::format::x;
        case KernelSelctor::xb: return cldnn::format::xb;
        case KernelSelctor::bx: return cldnn::format::bx;
        case KernelSelctor::yxfn: return cldnn::format::yxfn;
        case KernelSelctor::yxfb: return cldnn::format::yxfb;
        case KernelSelctor::byxf: return cldnn::format::byxf;
        case KernelSelctor::bfyx: return cldnn::format::bfyx;
        case KernelSelctor::fyxb: return cldnn::format::fyxb;
        case KernelSelctor::bs_xs_xsv8_bsv8: return cldnn::format::bs_xs_xsv8_bsv8;
        default:
            assert(0);
            return cldnn::format::x;
        }
    }

    jit_constants IGKConvolutionKernelBase::get_jit_constants(const ConvolutionParams& params, DispatchData kd) const
    {
        const auto split = 1; // TODO: do we need to support split (from performance aspect)?

        const int batch_size = params.outDims.w;
        const auto& cp = params.convParams;

        cldnn::tensor stride(
            cldnn::format::yx, 
            { (cldnn::tensor::value_type)std::min(cp.stride.y, params.inDims.y),
              (cldnn::tensor::value_type)std::min(cp.stride.x, params.inDims.x) });
        cldnn::tensor filter_tensor = cldnn::tensor(
            cldnn::format::os_iyx_osv16, // TODO: support more layouts
            { (cldnn::tensor::value_type)params.outDims.z,
              (cldnn::tensor::value_type)params.inDims.z,
              (cldnn::tensor::value_type)cp.filterSize.y,
              (cldnn::tensor::value_type)cp.filterSize.x });
        cldnn::tensor input_tensor = dim_2_tensor(params.inDims, params_2_cldnn(params.inputLayout));
        cldnn::tensor output_tensor = dim_2_tensor(params.outDims, params_2_cldnn(params.outputLayout));
        cldnn::tensor input_padding_tensor = cldnn::tensor(
            cldnn::format::yx,
            { (cldnn::tensor::value_type)cp.padding.y,
              (cldnn::tensor::value_type)cp.padding.x });
        cldnn::tensor output_padding_tensor = cldnn::tensor(
            cldnn::format::yx,
            { (cldnn::tensor::value_type)0,
              (cldnn::tensor::value_type)0 });
        auto input_offset_with_padding = params.inDesc.offset - cp.padding.x - params.inDesc.pitches.x*cp.padding.y;

        jit_constants mem_consts{
            neural::gpu::make_jit_constant("INPUT",                     input_tensor),
            neural::gpu::make_jit_constant("OUTPUT",                    output_tensor),
            neural::gpu::make_jit_constant("STRIDE",                    stride),
            neural::gpu::make_jit_constant("INPUT_OFFSET",              params.inDesc.offset), // should contains data for split also
            neural::gpu::make_jit_constant("INPUT_OFFSET_WITH_PADDING", input_offset_with_padding),
            neural::gpu::make_jit_constant("OUTPUT_OFFSET",             params.outDesc.offset),
            neural::gpu::make_jit_constant("OUTPUT_LIMIT",              params.outDesc.pitches.w),
            neural::gpu::make_jit_constant("INPUT_PADDING",             input_padding_tensor),
            neural::gpu::make_jit_constant("OUTPUT_PADDING",            output_padding_tensor),
            neural::gpu::make_jit_constant("FILTER",                    filter_tensor),
            neural::gpu::make_jit_constant("FILTER_ARRAY_NUM",          split),
            neural::gpu::make_jit_constant("FILTER_OUTPUT_FEATURE_NUM", "FILTER_FEATURE_NUM_0"),
            neural::gpu::make_jit_constant("FILTER_INPUT_FEATURE_NUM",  "FILTER_FEATURE_NUM_1"),
            neural::gpu::make_jit_constant("FP16_SUPPORTED",            static_cast<int>(kd.fp16_unit_used)), // TODO: why do we need it? FP16_UNIT_USED is enough.
            neural::gpu::make_jit_constant("FP16_UNIT_USED",            static_cast<int>(kd.fp16_unit_used)),
            neural::gpu::make_jit_constant("UNIT_TYPE",                 kd.fp16_unit_used ? "half" : "float"),
            neural::gpu::make_jit_constant("UNIT_VAL_ZERO",             kd.fp16_unit_used ? "0.0h" : "0.0f"),
            neural::gpu::make_jit_constant("RELU",                      static_cast<int>(params.activationFunc == ActivationFunction::RELU)),
            neural::gpu::make_jit_constant("NEGATIVE_SLOPE",            0.f), // TODO - add it to params
            neural::gpu::make_jit_constant("INPUT_ROW_PITCH",           params.inDesc.pitches.x),
            neural::gpu::make_jit_constant("INPUT_SLICE_PITCH",         params.inDesc.pitches.y),
            neural::gpu::make_jit_constant("INPUT_BATCH_PITCH",         params.inDesc.pitches.z),
            neural::gpu::make_jit_constant("OUT_ROW_PITCH",             params.outDesc.pitches.x),
            neural::gpu::make_jit_constant("OUT_SLICE_PITCH",           params.outDesc.pitches.y),
            neural::gpu::make_jit_constant("OUT_BATCH_PITCH",           params.outDesc.pitches.z),
        };

#if 0 // TODO when we will add batch kernels we need to support it
        if (filter_mem.argument().format == memory::format::yxio_f32 ||
            filter_mem.argument().format == memory::format::yxoi_f32 ||
            filter_mem.argument().format == memory::format::yxio_f16)
        {
            const auto local_work_group_size = kd.lws0;

            mem_consts.add_constant(neural::gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE", local_work_group_size));
            mem_consts.add_constant(neural::gpu::make_jit_constant("OFM_PER_WORK_ITEM", kd.ofm_per_work_item)); // how many output feature maps for a single batch will a single work item produce
            mem_consts.add_constant(neural::gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", kd.batches_per_work_item)); // how many batches will a single work item compute
            mem_consts.add_constant(neural::gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max(batch_size / kd.batches_per_work_item / local_work_group_size, static_cast<size_t>(1)))); // how many local work groups we need to compute single element for each batch
            mem_consts.add_constant(neural::gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS", batch_size / kd.batches_per_work_item)); // how many work items we need to compute single element for each batch

            if (params.inDims.z > 4)
            {
                mem_consts.add_constant(neural::gpu::make_jit_constant("USE_BLOCK_READ_2", ""));
            }
        }
#endif

        if (params.inputLayout == bfyx)
        {
            mem_consts.add_constant(neural::gpu::make_jit_constant("SUB_GROUP_SIZE", kd.lws2));
            mem_consts.add_constant(neural::gpu::make_jit_constant("OUT_BLOCK_WIDTH", kd.block_width));
            mem_consts.add_constant(neural::gpu::make_jit_constant("OUT_BLOCK_HEIGHT", kd.block_height));
            mem_consts.add_constant(neural::gpu::make_jit_constant("IN_BLOCK_ARRAY_SIZE", kd.input_block_array_size));
            mem_consts.add_constant(neural::gpu::make_jit_constant("IN_BLOCK_WIDTH", kd.input_block_width));
            mem_consts.add_constant(neural::gpu::make_jit_constant("PREFETCH", kd.prefetch));
            if (kd.leftovers)
                mem_consts.add_constant(neural::gpu::make_jit_constant("LEFTOVERS", kd.leftovers));
        }

        return mem_consts;
    }

    DispatchData IGKConvolutionKernelBase::set_default(const ConvolutionParams& params) const
    {
        auto split = 1;  // TODO: do we need to support split (from performance aspect)?
        auto batch_size = params.outDims.w;
        auto input_fetures = params.outDims.z;

        DispatchData kd;

        kd.fp16_unit_used = params.inputType == Datatype::F16;
        std::size_t gws0 = (input_fetures * batch_size) / split;
        std::size_t lws0 = std::min(gws0, static_cast<size_t>(32));
        while (gws0 % lws0)
        {
            lws0--;
        }
        kd.gws1 = params.outDims.x;
        kd.gws2 = params.outDims.y;
        kd.lws1 = kd.lws2 = 1;
        kd.ofm_per_work_item = 1;
        kd.batches_per_work_item = 1;
        kd.block_width = 1;
        kd.block_height = 1;
        kd.prefetch = 0;
        kd.input_block_array_size = 0;
        kd.input_block_width = 0;
        kd.leftovers = 0;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }
}