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

#include "fully_connected_gpu_bx_bs_x_bsv16_b1.h"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{
    ParamsKey FullyConnected_bx_bs_x_bsv16_b1::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F16);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetInputLayout(DataLayout::bf);
        k.SetOutputLayout(DataLayout::bf);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        return k;
    }

    FullyConnected_bx_bs_x_bsv16_b1::DispatchData FullyConnected_bx_bs_x_bsv16_b1::default_bfyx_bs_x_bsv16_b1(const FullyConnectedParams& arg) const
    {
        DispatchData run_info = set_kernel_data(arg);

        const auto response_size = arg.output.feature().v;

        // Properties of chunk and unit.
        const uint32_t unit_byte_size = run_info.fp16_unit_used ? sizeof(cl_half) : sizeof(float);
        const char* chunk_type = "uint";
        constexpr uint32_t chunk_byte_size = sizeof(cl_uint);
        constexpr uint32_t sub_group_size = 16;
        const uint32_t units_per_chunk = chunk_byte_size / unit_byte_size;
        const uint32_t units_per_sg_read = sub_group_size * units_per_chunk;
        // Properties of primitive responses.
        constexpr uint32_t responses_per_sg_exec = 16; // Must match batch slice size of weights format (bs_x_bsv16).
                                                       // Number of response groups. Each group (except last) writes responses_per_sg_exec responses
                                                       // for at least one input data set from batch.
        auto rg_count = cldnn::ceil_div(response_size, responses_per_sg_exec);

        run_info.lws0 = sub_group_size;
        // Number of work items needed to process all response groups.
        run_info.gws0 = rg_count * sub_group_size;
        run_info.lws1 = run_info.lws2 = 1;
        run_info.gws1 = run_info.gws2 = 1;

        //run_info.kernel_name = kernel_name_bx_bs_x_bsv16_b1;

        run_info.data_bx_bs_x_bsv16.unit_byte_size = unit_byte_size;
        run_info.data_bx_bs_x_bsv16.chunk_type = chunk_type;
        run_info.data_bx_bs_x_bsv16.chunk_byte_size = chunk_byte_size;
        run_info.data_bx_bs_x_bsv16.units_per_chunk = units_per_chunk;
        run_info.data_bx_bs_x_bsv16.bytes_per_sg_read = sub_group_size * chunk_byte_size;
        run_info.data_bx_bs_x_bsv16.units_per_sg_read = units_per_sg_read;
        run_info.data_bx_bs_x_bsv16.responses_per_sg_exec = responses_per_sg_exec;
        run_info.data_bx_bs_x_bsv16.in_chunk_prefetch_size = 2;
        run_info.data_bx_bs_x_bsv16.filter_chunk_prefetch_size = responses_per_sg_exec;

        return run_info;
    }

    KernelsData FullyConnected_bx_bs_x_bsv16_b1::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        const FullyConnectedParams& orgParams = static_cast<const FullyConnectedParams&>(params);

        const bool bSupportedActivation =
            orgParams.activationFunc == ActivationFunction::NONE ||
            orgParams.activationFunc == ActivationFunction::RELU ||
            orgParams.activationFunc == ActivationFunction::RELU_NEGATIVE_SLOPE;
        
        if (!bSupportedActivation)
        {
            return KernelsData();
        }

        KernelData kd;

        auto params_ptr = std::make_shared<FullyConnectedParams>(orgParams);
        kd.params = params_ptr;

        FullyConnectedParams& newParams = *params_ptr.get();
        
        kd.kernels.resize(1);
        DispatchData run_info;
        std::string jit;
        
        try
        {
            run_info = default_bfyx_bs_x_bsv16_b1(newParams);
            auto cldnn_jit = get_jit_constants(newParams, run_info);
            jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), kernel_name);
        }
        catch (const std::runtime_error& )
        {
            return KernelsData();
        }

        auto& kernel = kd.kernels[0];
        fill_cl_kernel_data(kernel, run_info, kernel_name, jit, orgParams.kernelID, true, !orgParams.bias.empty());

        auto cpu_kernel = CPUIGKFullyConnectedReorder(
            CPUIGKFullyConnectedReorder::WeightsReorderLayout::oiyx,
            CPUIGKFullyConnectedReorder::WeightsReorderLayout::os_i_osv16,
            params_ptr,
            run_info);

        kd.weights_reorder_params.engine = WeightsReorderParams::Engine::CPU;
        kd.weights_reorder_params.cpu_kernel = std::make_shared<CPUIGKFullyConnectedReorder>(cpu_kernel);
        kd.weights_reorder_params.new_buffer_size = cpu_kernel.GetNewWeightBufferSizeInBytes();

        kd.estimated_time = FORCE_PRIORITY_3;

        return{ kd };
    }
}