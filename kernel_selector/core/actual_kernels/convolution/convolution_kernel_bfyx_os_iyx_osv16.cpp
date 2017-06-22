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
#include "kernel_selector_utils.h"
#include "api/CPP/cldnn_defs.h"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_bfyx_os_iyx_osv16::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetInputWeightsType(WeightsType::F16);
        k.SetInputWeightsType(WeightsType::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetOutputLayout(DataLayout::bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetSubGroupSupport();
        k.SetBiasPerFeatureMap();
        k.SetBiasPerOutput();
        k.SetNonBiasSupport();
        k.SetBatchingSupport();
        k.SetSplitSupport();
        k.SetDilationSupport();
        return k;
    }

    static std::pair<size_t, size_t> get_bfyx_req_input_block_dims(
        size_t output_block_width,
        size_t output_block_height,
        const uSize& filter_size,
        const uSize& stride,
        const uSize& dilation,
        size_t sub_group_size = 16,
        size_t read_chunk_size = 8,
        size_t min_read_size = 16)
    {
        assert(output_block_width > 0 && output_block_height > 0);
        assert(stride.x > 0 && stride.y > 0);
        assert(filter_size.x > 0 && filter_size.y > 0);

        // Number of elements in X dimension needed from input to compute output block without re-reading input.
        size_t input_block_req_width = (output_block_width - 1) * stride.x + (filter_size.x - 1)*dilation.x + 1;
        // Number of elements in Y dimension needed from input to compute output block without re-reading input.
        size_t input_block_req_height = (output_block_height - 1) * stride.y + (filter_size.y - 1)*dilation.y + 1;

        // Required number of elements in X dimension rounded to nearest >= read chunk size.
        size_t input_block_read_width = std::max(cldnn::round_up_to(input_block_req_width, read_chunk_size), min_read_size);
        // Number of sub-group-sized vectors of unit type needed to store input block.
        size_t input_block_array_size = cldnn::ceil_div(input_block_req_height * input_block_read_width, sub_group_size);

        return std::make_pair(input_block_array_size, input_block_read_width);
    }

    IGKConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_os_iyx_osv16::default_bfyx_os_iyx_osv16(const ConvolutionParams& arg) const
    {
        DispatchData runInfo = SetDefault(arg);

        // Maximum supported size (in any dimension) of filter by "kernel_name_bfyx_os_iyx_osv16" kernel.
        constexpr size_t max_supported_filter_size = 11;
        // Sub-group size used by "kernel_name_bfyx_os_iyx_osv16" kernel.
        constexpr size_t sub_group_size = 16;

        const auto of_maps = arg.output.Feature().v;
        const size_t of_threads_per_batch = cldnn::round_up_to(of_maps, sub_group_size);
        runInfo.leftovers = of_threads_per_batch - of_maps;

        const auto cp = arg.convParams;
        if (cp.filterSize.x > max_supported_filter_size ||
            cp.filterSize.y > max_supported_filter_size)
        {
            throw std::runtime_error("Unsupported filter size (> 11) in bfyx convolution");
        }

        runInfo.effiency = FORCE_PRIORITY_3;

        if (cp.stride.x == 1 && cp.stride.y == 1)
        {
            if (cp.filterSize.x == 1 && cp.filterSize.y == 1)
            {
                runInfo.blockWidth = 16;
                runInfo.blockHeight = 1;
                runInfo.prefetch = 4;
            }
            //if less than 16 values is required to compute one single row of output
            //then each WI shall compute one single row to maximize reuse within SIMD subgroup (this gives very nice performance results)
            else if (arg.output.X().v + (cp.filterSize.x - 1)*cp.dilation.x < sub_group_size)
            {
                runInfo.blockWidth = arg.output.X().v;
                runInfo.blockHeight = 1;
                runInfo.prefetch = 4;
            }
            else if (cp.filterSize.x < 5 && cp.filterSize.y < 5)
            {
                runInfo.blockWidth = sub_group_size - cp.filterSize.x + 1;
                runInfo.blockHeight = 2;
                runInfo.prefetch = 4;
            }
            else
            {
                runInfo.blockWidth = 4;
                runInfo.blockHeight = 3;
                runInfo.prefetch = 4;
            }
        }
        else if (cp.stride.x == 2 && cp.stride.y == 2)
        {
            runInfo.blockWidth = 5;
            runInfo.blockHeight = 4;
            runInfo.prefetch = 4;
        }
        else
        {
            runInfo.blockWidth = 4;
            runInfo.blockHeight = 3;
            runInfo.prefetch = 5;
            //run_info.effiency = FORCE_PRIORITY_7; // GEMM is better
        }


        auto input_block_dims = get_bfyx_req_input_block_dims(
            runInfo.blockWidth, 
            runInfo.blockHeight,
            cp.filterSize,
            cp.stride,
            cp.dilation,
            sub_group_size,
            runInfo.fp16UnitUsed ? sub_group_size : sub_group_size / 2,
            sub_group_size);
        runInfo.inputBlockArraySize = input_block_dims.first;
        runInfo.inputBlockWidth = input_block_dims.second;

        runInfo.gws0 = cldnn::ceil_div(arg.output.X().v, runInfo.blockWidth);
        runInfo.gws1 = cldnn::ceil_div(arg.output.Y().v, runInfo.blockHeight);
        runInfo.gws2 = of_threads_per_batch * arg.output.Batch().v;

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = sub_group_size;

        return runInfo;
    }

    KernelsData ConvolutionKernel_bfyx_os_iyx_osv16::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);
        const bool bSupportedWeightsLayout = orgParams.weights.GetLayout() == WeightsLayout::oiyx;
        const bool bWeightsOK = bSupportedWeightsLayout || optParams.allowWeightsReorder;
        const auto req_input = GetConvolutionPaddedTensorDesc(orgParams);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(orgParams, req_input);
        const bool bInputPadded = optParams.allowPadding || bProperInputDesc;
        const bool bSupportedActivation = CheckActivationSupport(orgParams.activationFunc);
        
        if (!bInputPadded || !bSupportedActivation || !bWeightsOK)
        {
            return KernelsData();
        }

        KernelData kd;

        auto params_ptr = std::make_shared<ConvolutionParams>(orgParams);
        kd.params = params_ptr;

        ConvolutionParams& newParams = *params_ptr.get();
        
        kd.kernels.resize(1);
        DispatchData runInfo;
        
        try
        {
            runInfo = default_bfyx_os_iyx_osv16(newParams);
        }
        catch (const std::runtime_error& )
        {
            return KernelsData();
        }
        
        if (optParams.allowPadding && !bProperInputDesc)
        {
            newParams.inputs[0] = req_input;
            kd.reorderInput = true;
        }

        auto cldnn_jit = GetJitConstants(newParams, runInfo);
        cldnn_jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws2));
        cldnn_jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_WIDTH", runInfo.blockWidth));
        cldnn_jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_HEIGHT", runInfo.blockHeight));
        cldnn_jit.AddConstant(MakeJitConstant("IN_BLOCK_ARRAY_SIZE", runInfo.inputBlockArraySize));
        cldnn_jit.AddConstant(MakeJitConstant("IN_BLOCK_WIDTH", runInfo.inputBlockWidth));
        cldnn_jit.AddConstant(MakeJitConstant("PREFETCH", runInfo.prefetch));
        if (runInfo.leftovers)
        {
            cldnn_jit.AddConstant(MakeJitConstant("LEFTOVERS", runInfo.leftovers));
        }
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, true, !orgParams.bias.empty());
        kernel.argsDesc.data.push_back({ ArgumentDescpirtor::Types::SPLIT, 0 });

        bool succeed = SetWeightsReorderParams(newParams, WeightsLayout::os_iyx_osv16, kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}