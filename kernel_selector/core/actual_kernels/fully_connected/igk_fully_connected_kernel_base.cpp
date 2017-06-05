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
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{
    void IGKFullyConnectedKernelBase::CPUIGKFullyConnectedReorder::Execute(void* input, size_t, void* output, size_t) const
    {
        assert(input_layout == WeightsReorderLayout::oiyx);
        assert(output_layout == WeightsReorderLayout::os_i_osv16);
        const auto batch_size = params->output.batch().v;
        const auto ifms = params->inputs[0].feature();
        const auto x_size = ifms.pitch * ifms.v;
        const auto weights_line = params->inputs[0].Length() / batch_size;

        short* input_ptr = (short*)input;
        short* output_ptr = (short*)output;

        const auto ofms = params->output.feature();
        const auto input_x = params->inputs[0].x();
        const auto input_y = params->inputs[0].y();

        for (size_t ofm = 0; ofm < ofms.v; ofm++)
        {
            for (size_t x = 0; x < x_size; x++)
            {
                const size_t _slice_id = ofm / 16;
                const size_t _id_in_slice = ofm % 16;
                const size_t output_idx = _id_in_slice + 16 * (x + x_size * _slice_id);

                if ((x % input_y.pitch) < input_x.v)
                {
                    // TODO: handle pitches
                    const size_t in_x = x % input_y.pitch;
                    const size_t in_other = x / input_y.pitch;
                    const size_t in_final_x = in_x + in_other * input_x.v;
                    size_t input_idx = in_final_x + weights_line * ofm;

                    output_ptr[output_idx] = input_ptr[input_idx];
                }
                else
                {
                    output_ptr[output_idx] = 0;
                }
            }
        }
    }

    size_t IGKFullyConnectedKernelBase::CPUIGKFullyConnectedReorder::GetNewWeightBufferSizeInBytes() const
    {
        constexpr uint32_t sub_group_size = 16;
        const auto ifm = params->inputs[0].feature();
        const auto x_size = ifm.pitch * ifm.v;
        const auto ofm = params->output.feature().v;
        return
            x_size *
            cldnn::round_up_to(ofm, sub_group_size) *
            BytesPerElement(params->inputs[0].dtype);
    }

    jit_constants IGKFullyConnectedKernelBase::get_jit_constants(const FullyConnectedParams& params, IGKFullyConnectedKernelBase::DispatchData data) const
    {
        const auto ifm = params.inputs[0].feature();
        const auto x_size = ifm.pitch * ifm.v;

        cldnn::tensor filter_tensor = cldnn::tensor(
              (cldnn::tensor::value_type)params.output.feature().v,
              (cldnn::tensor::value_type)x_size,
              (cldnn::tensor::value_type)1,
              (cldnn::tensor::value_type)1);

        cldnn::tensor input_tensor = ks_tensor_2_tensor(params.inputs[0]);
        cldnn::tensor output_tensor = ks_tensor_2_tensor(params.output);

        const bool relu =
            params.activationFunc == ActivationFunction::RELU ||
            params.activationFunc == ActivationFunction::RELU_NEGATIVE_SLOPE;
        const float negative_slope =
            params.activationFunc == ActivationFunction::RELU_NEGATIVE_SLOPE ?
            params.nlParams.m : 0.f;

        jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",                     input_tensor),
            gpu::make_jit_constant("OUTPUT",                    output_tensor),
            gpu::make_jit_constant("INPUT_OFFSET",              params.inputs[0].offset),
            gpu::make_jit_constant("OUTPUT_OFFSET",             params.output.offset),
            gpu::make_jit_constant("INPUT_ELEMENTS_COUNT",      x_size),
            gpu::make_jit_constant("WEIGHTS",                   filter_tensor),
            gpu::make_jit_constant("FP16_SUPPORTED",            static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("FP16_UNIT_USED",            static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",                 data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",             data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("RELU",                      static_cast<int>(relu)),
            gpu::make_jit_constant("NEGATIVE_SLOPE",            negative_slope),
            // TODO handle layout
//             gpu::make_jit_constant("INPUT_Y_PITCH",           params.inDesc.pitches.x),
//             gpu::make_jit_constant("INPUT_FEATURE_PITCH",         params.inDesc.pitches.y),
//             gpu::make_jit_constant("INPUT_BATCH_PITCH",         params.inDesc.pitches.z),
//             gpu::make_jit_constant("OUT_Y_PITCH",             params.outDesc.pitches.x),
//             gpu::make_jit_constant("OUT_FEATURE_PITCH",           params.outDesc.pitches.y),
//             gpu::make_jit_constant("OUT_BATCH_PITCH",           params.outDesc.pitches.z),
        };

#if 0
        if (data.kernel_name == kernel_name_xb_xb_block_fp16)
        {
            mem_consts.add_constant(gpu::make_jit_constant("SUB_GROUP_SIZE", data.lws0));
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_BATCH", data.gws1));

            mem_consts.add_constant(gpu::make_jit_constant("UNIT_BYTE_SIZE", data.data_xb_xb_fp16.unit_byte_size));
            mem_consts.add_constant(gpu::make_jit_constant("CHUNK_TYPE", data.data_xb_xb_fp16.chunk_type));
            mem_consts.add_constant(gpu::make_jit_constant("CHUNK_BYTE_SIZE", data.data_xb_xb_fp16.chunk_byte_size));
            mem_consts.add_constant(gpu::make_jit_constant("UNITS_PER_CHUNK", data.data_xb_xb_fp16.units_per_chunk));
            mem_consts.add_constant(gpu::make_jit_constant("BYTES_PER_SG_READ", data.data_xb_xb_fp16.bytes_per_sg_read));
            mem_consts.add_constant(gpu::make_jit_constant("UNITS_PER_SG_READ", data.data_xb_xb_fp16.units_per_sg_read));
            mem_consts.add_constant(gpu::make_jit_constant("RG_COUNT", data.data_xb_xb_fp16.rg_count));
            mem_consts.add_constant(gpu::make_jit_constant("LAST_RG_SIZE", data.data_xb_xb_fp16.last_rg_size));
        }
#endif
        //if (data.kernel_name == kernel_name_bx_bs_x_bsv16_b1)
        {
            mem_consts.add_constant(gpu::make_jit_constant("SUB_GROUP_SIZE", data.lws0));
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_BATCH", data.gws1));

            mem_consts.add_constant(gpu::make_jit_constant("UNIT_BYTE_SIZE", data.data_bx_bs_x_bsv16.unit_byte_size));
            mem_consts.add_constant(gpu::make_jit_constant("CHUNK_TYPE", data.data_bx_bs_x_bsv16.chunk_type));
            mem_consts.add_constant(gpu::make_jit_constant("CHUNK_BYTE_SIZE", data.data_bx_bs_x_bsv16.chunk_byte_size));
            mem_consts.add_constant(gpu::make_jit_constant("UNITS_PER_CHUNK", data.data_bx_bs_x_bsv16.units_per_chunk));
            mem_consts.add_constant(gpu::make_jit_constant("BYTES_PER_SG_READ", data.data_bx_bs_x_bsv16.bytes_per_sg_read));
            mem_consts.add_constant(gpu::make_jit_constant("UNITS_PER_SG_READ", data.data_bx_bs_x_bsv16.units_per_sg_read));
            mem_consts.add_constant(gpu::make_jit_constant("RESPONSES_PER_SG_EXEC", data.data_bx_bs_x_bsv16.responses_per_sg_exec));
            mem_consts.add_constant(gpu::make_jit_constant("IN_CHUNK_PREFETCH_SIZE", data.data_bx_bs_x_bsv16.in_chunk_prefetch_size));
            mem_consts.add_constant(gpu::make_jit_constant("FILTER_CHUNK_PREFETCH_SIZE", data.data_bx_bs_x_bsv16.filter_chunk_prefetch_size));
        }

#if 0
        if (data.kernel_name == kernel_name_xb_xb_b8_x8_vload ||
            data.kernel_name == kernel_name_xb_bs_xs_xsv8_bsv8_vload)
        {
            const int batches_per_work_item = get_batches_per_work_item(output_mem);

            mem_consts.add_constant(gpu::make_jit_constant("NEURONS_PER_WORK_ITEM", get_neurons_per_work_item(output_mem))); // how many neurons for a single batch will a single work item produce
            mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", batches_per_work_item));                 // how many batches will a single work item compute
            mem_consts.add_constant(gpu::make_jit_constant("OUTPUT_ELEMENTS_COUNT", output_mem.count() / output_mem.get_layout().size.batch[0]));
        }
#endif
        return mem_consts;
    }

    IGKFullyConnectedKernelBase::DispatchData IGKFullyConnectedKernelBase::set_kernel_data(const FullyConnectedParams& params) const
    {
        DispatchData kd;

        kd.fp16_unit_used = params.inputs[0].dtype == Datatype::F16;

        // Determine global work sizes.
        kd.gws0 = params.output.Length();
        kd.gws1 = 1;

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;

        return kd;
    }
}