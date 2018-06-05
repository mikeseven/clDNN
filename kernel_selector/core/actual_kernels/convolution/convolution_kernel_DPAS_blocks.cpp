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

#include "convolution_kernel_DPAS_blocks.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {
    
    ParamsKey ConvolutionKernel_DPAS_blocks::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableInputWeightsType(WeightsType::INT8);
        k.EnableInputLayout(DataLayout::byxf_af32);
        k.EnableOutputLayout(DataLayout::byxf_af32);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDepthwiseSeparableOpt();
        k.EnableInt8Quantization();
        k.EnableOutputCalibration();
        k.DisableTuning();
        return k;
    }

    ConvolutionKernel_DPAS_blocks::BlockSizes ConvolutionKernel_DPAS_blocks::getOutputBlockSizes(const Params& p) const
    {
        BlockSizes bs = { 0, 0, 0 };

        constexpr size_t sub_group_size = 8;

        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);
        const auto cp = params.convParams;

        if (cp.stride.x == 1 && cp.stride.y == 1)
        {
            if (cp.filterSize.x == 1 && cp.filterSize.y == 1)
            {
                bs.blockWidth = 8;
                bs.blockHeight = 1;
                bs.prefetch = 4;
            }
            //if less than 8 values is required to compute one single row of output
            //then each WI shall compute one single row to maximize reuse within SIMD subgroup (this gives very nice performance results)
            else if (params.output.X().v + (cp.filterSize.x - 1)*cp.dilation.x < sub_group_size)
            {
                bs.blockWidth = params.output.X().v;
                bs.blockHeight = 1;
                bs.prefetch = 4;
            }
            else if (cp.filterSize.x < 5 && cp.filterSize.y < 5)
            {
                bs.blockWidth = sub_group_size - cp.filterSize.x + 1;
                bs.blockHeight = 2;
                bs.prefetch = 4;
            }
            else
            {
                bs.blockWidth = 4;
                bs.blockHeight = 3;
                bs.prefetch = 4;
            }
        }
        else if (cp.stride.x == 2 && cp.stride.y == 2)
        {
            bs.blockWidth = 5;
            bs.blockHeight = 4;
            bs.prefetch = 4;
        }
        else
        {
            bs.blockWidth = 4;
            bs.blockHeight = 3;
            bs.prefetch = 5;
            //run_info.effiency = FORCE_PRIORITY_7; // GEMM is better
        }
/*        bs.blockWidth = 2;
        bs.blockHeight = 1;*/
        return bs;
    }

    static std::pair<size_t, size_t> get_byxf_af32_req_input_block_dims(
        size_t output_block_width,
        size_t output_block_height,
        const uSize& filter_size,
        const uSize& stride,
        const uSize& dilation,
        size_t sub_group_size = 8,
        size_t read_chunk_size = 8,
        size_t min_read_size = 8)
    {
        assert(output_block_width > 0 && output_block_height > 0);
        assert(stride.x > 0 && stride.y > 0);
        assert(filter_size.x > 0 && filter_size.y > 0);

        // Number of elements in X dimension needed from input to compute output block without re-reading input.
        size_t input_block_req_width = (output_block_width - 1) * stride.x + (filter_size.x - 1)*dilation.x + 1;
        // Number of elements in Y dimension needed from input to compute output block without re-reading input.
        size_t input_block_req_height = (output_block_height - 1) * stride.y + (filter_size.y - 1)*dilation.y + 1;

        // Required number of elements in X dimension rounded to nearest >= read chunk size.
        size_t input_block_read_width = std::max(RoundUp(input_block_req_width, read_chunk_size), min_read_size);
        // Number of sub-group-sized vectors of unit type needed to store input block.
        size_t input_block_array_size = CeilDiv(input_block_req_height * input_block_read_width, sub_group_size);

        return std::make_pair(input_block_array_size, input_block_read_width);
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_DPAS_blocks::SetDefault(const ConvolutionParams& arg, int) const
    {
        // Sub-group size used by "kernel_name_bfyx_os_iyx_osv16" kernel.
        constexpr size_t sub_group_size = 8;

        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

        const auto cp = arg.convParams;

        auto blockSizes = getOutputBlockSizes(arg);
        runInfo.cldnnStyle.blockWidth = blockSizes.blockWidth;
        runInfo.cldnnStyle.blockHeight = blockSizes.blockHeight;
        runInfo.cldnnStyle.prefetch = blockSizes.prefetch;

        auto input_block_dims = get_byxf_af32_req_input_block_dims(
            runInfo.cldnnStyle.blockWidth,
            runInfo.cldnnStyle.blockHeight,
            cp.filterSize,
            cp.stride,
            cp.dilation,
            sub_group_size,
            runInfo.fp16UnitUsed ? sub_group_size : sub_group_size / 2,
            sub_group_size);
        runInfo.cldnnStyle.inputBlockArraySize = input_block_dims.first;
        runInfo.cldnnStyle.inputBlockWidth = input_block_dims.second;


        const auto of_maps = arg.output.Feature().v;
        const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);
        runInfo.cldnnStyle.leftovers = of_threads_per_batch - of_maps;

        runInfo.effiency = FORCE_PRIORITY_3;

        runInfo.gws0 = CeilDiv(arg.output.X().v, runInfo.cldnnStyle.blockWidth);
        runInfo.gws1 = CeilDiv(arg.output.Y().v, runInfo.cldnnStyle.blockHeight);
        runInfo.gws2 = of_threads_per_batch * arg.output.Batch().v;

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = sub_group_size;

        return runInfo;
    }

    JitConstants ConvolutionKernel_DPAS_blocks::GetJitConstants(const ConvolutionParams& params, DispatchData runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws2));
        jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_WIDTH", runInfo.cldnnStyle.blockWidth));
        jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_HEIGHT", runInfo.cldnnStyle.blockHeight));
        jit.AddConstant(MakeJitConstant("IN_BLOCK_ARRAY_SIZE", runInfo.cldnnStyle.inputBlockArraySize));
        jit.AddConstant(MakeJitConstant("IN_BLOCK_WIDTH", runInfo.cldnnStyle.inputBlockWidth));
        jit.AddConstant(MakeJitConstant("PREFETCH", runInfo.cldnnStyle.prefetch));

        return jit;
    }

    KernelsData ConvolutionKernel_DPAS_blocks::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        KernelsData kd = GetCommonKernelsData(params, options);
        kd[0].estimatedTime = FORCE_PRIORITY_1;
        return kd;//return GetCommonKernelsData(params, options);
    }
}