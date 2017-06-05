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

#pragma once

#include "igk_kernel_base.h"
#include "kernel_selector_params.h"

namespace KernelSelector 
{
    class IGKFullyConnectedKernelBase : public IGKKernelBase
    {
    public:
        using IGKKernelBase::IGKKernelBase;
        virtual ~IGKFullyConnectedKernelBase() {}

        struct DispatchData
        {
            //std::vector<cldnn::refcounted_obj_ptr<cldnn::network_impl>> reorder;
            size_t gws0, gws1;
            size_t lws0, lws1;
            //std::string kernel_name;
            bool fp16_unit_used;
            union
            {
                struct
                {
                    uint32_t unit_byte_size;
                    const char* chunk_type;
                    uint32_t chunk_byte_size;
                    uint32_t units_per_chunk;
                    uint32_t bytes_per_sg_read;
                    uint32_t units_per_sg_read;
                    uint32_t rg_count;
                    uint32_t last_rg_size;
                } data_xb_xb_fp16;
                struct
                {
                    uint32_t unit_byte_size;
                    const char* chunk_type;
                    uint32_t chunk_byte_size;
                    uint32_t units_per_chunk;
                    uint32_t bytes_per_sg_read;
                    uint32_t units_per_sg_read;
                    uint32_t responses_per_sg_exec;
                    uint32_t in_chunk_prefetch_size;
                    uint32_t filter_chunk_prefetch_size;
                } data_bx_bs_x_bsv16;
            };
        };
        struct CPUIGKFullyConnectedReorder : public CPUKernel
        {
            enum class WeightsReorderLayout
            {
                oiyx,
                yxoi,
                oyxi,
                yxio,
                os_iyx_osv16,
                os_i_osv16,
            };

            WeightsReorderLayout input_layout = WeightsReorderLayout::oiyx;
            WeightsReorderLayout output_layout = WeightsReorderLayout::oiyx;
            std::shared_ptr<FullyConnectedParams> params;
            DispatchData kd;

            CPUIGKFullyConnectedReorder(WeightsReorderLayout in_layout, WeightsReorderLayout out_layout, std::shared_ptr<FullyConnectedParams> _params, DispatchData kd) :
                input_layout(in_layout),
                output_layout(out_layout),
                params(_params),
                kd(kd) {}

            virtual void Execute(void* input, size_t input_size, void* output, size_t output_size) const;
            size_t GetNewWeightBufferSizeInBytes() const;
        };
    
    protected:
        jit_constants get_jit_constants(const FullyConnectedParams& params, DispatchData kd) const;
        DispatchData set_kernel_data(const FullyConnectedParams& params) const;
        const std::string weights_reorder_kernel_name = "cnn_align_weights";

        static std::string Float2Str(const float f)
        {
            return std::to_string(f) + "f";
        }

        std::string GetWeightsBaseJit(const BaseParams& params) const
        {
            std::stringstream jit;

            jit << "#define ACTIVATION_FUNCTION_" << toString(params.activationFunc) << "\n"
                << "#define TYPE_" << toString(params.inputs[0].dtype) << "\n"
                << "#define NL_M (" << Float2Str(params.nlParams.m) << ")\n"
                << "#define NL_N (" << Float2Str(params.nlParams.n) << ")\n"
                << "#define INPUT_OFFSET (" << params.inputs[0].offset << ")\n"
                << "#define OUT_OFFSET (" << params.output.offset << ")\n";

            jit << "#define INPUT_WIDTH (" << params.inputs[0].x().v << ")\n"
                << "#define INPUT_HEIGHT (" << params.inputs[0].y().v << ")\n"
                << "#define INPUT_DEPTH (" << params.inputs[0].feature().v << ")\n"
                << "#define INPUT_BATCH (" << params.inputs[0].batch().v << ")\n"
                << "#define INPUT_Y_PITCH (" << params.inputs[0].y().pitch << ")\n"
                << "#define INPUT_FEATURE_PITCH (" << params.inputs[0].feature().pitch << ")\n"
                << "#define INPUT_BATCH_PITCH (" << params.inputs[0].batch().pitch << ")\n";

            jit << "#define OUT_WIDTH (" << params.output.x().v << ")\n"
                << "#define OUT_HEIGHT (" << params.output.y().v << ")\n"
                << "#define OUT_DEPTH (" << params.output.feature().v << ")\n"
                << "#define OUT_BATCH (" << params.output.batch().v << ")\n"
                << "#define OUT_Y_PITCH (" << params.output.y().pitch << ")\n"
                << "#define OUT_FEATURE_PITCH (" << params.output.feature().pitch << ")\n"
                << "#define OUT_BATCH_PITCH (" << params.output.batch().pitch << ")\n";

            return jit.str();
        }
    };
}