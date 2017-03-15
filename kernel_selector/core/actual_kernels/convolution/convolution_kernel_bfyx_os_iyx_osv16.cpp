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

#include "convolution_kernel_bfyx_os_iyx_osv16.h"
#include "api/cldnn_defs.h"

namespace KernelSelctor 
{

    ParamsKey ConvolutionKernel_bfyx_os_iyx_osv16::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetInputLayout(bfyx);
        k.SetOutputLayout(bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetSubGroupSupport();
        k.SetBiasPerFeatureMap();
        k.SetBiasPerOutput();
        k.SetNumDims(4);
        return k;
    }

    static std::pair<size_t, size_t> get_bfyx_req_input_block_dims(
        size_t output_block_width,
        size_t output_block_height,
        const uSize& filter_size,
        const uSize& stride,
        size_t sub_group_size = 16,
        size_t read_chunk_size = 8,
        size_t min_read_size = 16)
    {
        assert(output_block_width > 0 && output_block_height > 0);
        assert(stride.x > 0 && stride.y > 0);
        assert(filter_size.x > 0 && filter_size.y > 0);

        // Number of elements in X dimension needed from input to compute output block without re-reading input.
        std::size_t input_block_req_width = (output_block_width - 1) * stride.x + filter_size.x;
        // Number of elements in Y dimension needed from input to compute output block without re-reading input.
        std::size_t input_block_req_height = (output_block_height - 1) * stride.y + filter_size.y;

        // Required number of elements in X dimension rounded to nearest >= read chunk size.
        std::size_t input_block_read_width = std::max(round_up_to(input_block_req_width, read_chunk_size), min_read_size);
        // Number of sub-group-sized vectors of unit type needed to store input block.
        std::size_t input_block_array_size = ceil_div(input_block_req_height * input_block_read_width, sub_group_size);

        return std::make_pair(input_block_array_size, input_block_read_width);
    }

    DispatchData ConvolutionKernel_bfyx_os_iyx_osv16::default_bfyx_os_iyx_osv16(const ConvolutionParams& arg) const
    {
        DispatchData run_info = set_default(arg);

        // Maximum supported size (in any dimension) of filter by "kernel_name_bfyx_os_iyx_osv16" kernel.
        constexpr size_t max_supported_filter_size = 11;
        // Sub-group size used by "kernel_name_bfyx_os_iyx_osv16" kernel.
        constexpr uint32_t sub_group_size = 16;

        const auto of_maps = arg.outDims.z;
        const uint32_t of_threads_per_batch = round_up_to(of_maps, sub_group_size);
        run_info.leftovers = of_threads_per_batch - of_maps;

        const auto cp = arg.convParams;
        if (cp.filterSize.x > max_supported_filter_size ||
            cp.filterSize.y > max_supported_filter_size)
        {
            throw std::runtime_error("Unsupported filter size (> 11) in bfyx convolution");
        }

        run_info.effiency = arg.inDims.x <= sub_group_size ? FORCE_PRIORITY_3 : FORCE_PRIORITY_5;

        if (cp.stride.x == 1 && cp.stride.y == 1)
        {
            if (cp.filterSize.x == 1 && cp.filterSize.y == 1)
            {
                run_info.block_width = 16;
                run_info.block_height = 1;
                run_info.prefetch = 4;
                run_info.effiency = FORCE_PRIORITY_3;
            }
            //if less than 16 values is required to compute one single row of output
            //then each WI shall compute one single row to maximize reuse within SIMD subgroup (this gives very nice performance results)
            else if (arg.outDims.x + cp.filterSize.x - 1 < sub_group_size)
            {
                run_info.block_width = arg.outDims.x;
                run_info.block_height = 1;
                run_info.prefetch = 4;
                run_info.effiency = FORCE_PRIORITY_3;
            }
            else if (cp.filterSize.x < 5 && cp.filterSize.y < 5)
            {
                run_info.block_width = sub_group_size - cp.filterSize.x + 1;
                run_info.block_height = 2;
                run_info.prefetch = 4;
            }
            else
            {
                run_info.block_width = 4;
                run_info.block_height = 3;
                run_info.prefetch = 4;
            }
        }
        else if (cp.stride.x == 2 && cp.stride.y == 2)
        {
            run_info.block_width = 5;
            run_info.block_height = 4;
            run_info.prefetch = 4;
        }
        else if (cp.stride.x == 4 && cp.stride.y == 4)
        {
            run_info.block_width = 4;
            run_info.block_height = 3;
            run_info.prefetch = 5;
            run_info.effiency = FORCE_PRIORITY_7;
        }
        else
        {
            throw std::runtime_error("Unsupported stride (!= 1,2,4) in bfyx convolution");
        }


        auto input_block_dims = get_bfyx_req_input_block_dims(
            run_info.block_width, 
            run_info.block_height,
            cp.filterSize,
            cp.stride,
            sub_group_size,
            run_info.fp16_unit_used ? sub_group_size : sub_group_size / 2,
            sub_group_size);
        run_info.input_block_array_size = input_block_dims.first;
        run_info.input_block_width = input_block_dims.second;

        run_info.gws0 = ceil_div(arg.outDims.x, run_info.block_width);
        run_info.gws1 = ceil_div(arg.outDims.y, run_info.block_height);
        run_info.gws2 = of_threads_per_batch * arg.outDims.w;

        run_info.lws0 = 1;
        run_info.lws1 = 1;
        run_info.lws2 = sub_group_size;

        return run_info;
    }

    KernelsData ConvolutionKernel_bfyx_os_iyx_osv16::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);
        const auto req_input = GetConvolutionPaddedTensorDesc(orgParams);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(orgParams, req_input);
        const bool bInputPadded = optParams.allow_padding || bProperInputDesc;
        const bool bSupportedActivation =
            orgParams.activationFunc == ActivationFunction::NONE ||
            orgParams.activationFunc == ActivationFunction::RELU;
        
        if (!bInputPadded || !bSupportedActivation)
        {
            return KernelsData();
        }

        KernelData kd;

        auto params_ptr = std::make_shared<ConvolutionParams>(orgParams);
        kd.params = params_ptr;

        ConvolutionParams& newParams = *params_ptr.get();
        
        kd.kernels.resize(1);
        DispatchData run_info;
        
        try
        {
            run_info = default_bfyx_os_iyx_osv16(newParams);
        }
        catch (const std::runtime_error& )
        {
            return KernelsData();
        }
        
        // for KW only
        kd.reorder_input = false;

        if (optParams.allow_padding && !bProperInputDesc)
        {
            newParams.inDesc = req_input;
            kd.reorder_input = true;
        }

        auto cldnn_jit = get_jit_constants(newParams, run_info);
        auto jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), kernel_name);

        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(run_info.gws0, run_info.gws1, run_info.gws2);
        kernel.work_groups.local = cl::NDRange(run_info.lws0, run_info.lws1, run_info.lws2);
        kernel.kernel_string = get_kernel_string(kernel_name, jit, kernel_name, ROUND_ROBIN);
        kernel.args_desc = get_args_desc(1, true, true);
        kernel.args_desc.data.push_back({ ArgumentDescpirtor::Types::UINT32, 0 });

        auto cpu_kernel = CPUIGKConvolutionReorder(
            CPUIGKConvolutionReorder::WeightsReorderLayout::oiyx,
            CPUIGKConvolutionReorder::WeightsReorderLayout::os_iyx_osv16,
            params_ptr,
            run_info);

        kd.weights_reorder_params.engine = WeightsReorderParams::Engine::CPU;
        kd.weights_reorder_params.cpu_kernel = std::make_shared<CPUIGKConvolutionReorder>(cpu_kernel);
        kd.weights_reorder_params.new_buffer_size = cpu_kernel.GetNewWeightBufferSizeInBytes();
        kd.estimated_time = run_info.effiency;

        return{ kd };
    }
}