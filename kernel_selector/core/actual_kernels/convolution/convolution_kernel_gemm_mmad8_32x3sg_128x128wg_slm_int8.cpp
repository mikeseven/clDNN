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

#include "convolution_kernel_gemm_mmad8_32x3sg_128x128wg_slm_int8.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
    
    ParamsKey convolution_kernel_gemm_mmad8_32x3sg_128x128wg_slm_int8::GetSupportedKey() const
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

    bool convolution_kernel_gemm_mmad8_32x3sg_128x128wg_slm_int8::Validate(const Params& p, const optional_params& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const convolution_params&>(p);

        if (params.filterSize.x != 1 || params.filterSize.y != 1)
            return false;

        if (params.stride.x != 1 || params.stride.y != 1)
            return false;

        if (params.padding.x != 0 || params.padding.y != 0)
            return false;

        const auto& input = params.inputs[0];
        const auto& output = params.output;

        // we do not support padded input
        if (input.X().pad.Total() != 0 || input.Y().pad.Total() != 0)
            return false;

        if (params.split != 1)
            return false;

        const auto m = input.X().v * input.Y().v *input.Batch().v;
        const auto k = input.Feature().v;
        const auto n = output.Feature().v * output.Batch().v;

        if (input.Batch().v != 64)
            return false;

        if (m % (128*8) != 0)
            return false;

        if (k % (32*8) != 0/* || output.Feature().v != 1024*/)
            return false;

        if (n % (128*8) != 0)
            return false;

        return true;
    }

#define WG_TILE_M 32  // Work-Group tile size M, Must be mutliple of 32
#define WG_TILE_N 32  // Work-Group tile size N, Must be mutliple of 32

#define DIM_X 0
#define DIM_Y 1
#define MATRIX_SMALL_K 32
#define SG_TILE_M 32
#define SG_TILE_N 32
#define SG_SIZE 8
#define SIMD_LANE_M SG_TILE_M
#define SIMD_LANE_N (SG_TILE_N / SG_SIZE)
#define WG_SIZE (SG_SIZE * WG_TILE_N / SG_TILE_N) * (WG_TILE_M / SG_TILE_M)

    ConvolutionKernelBase::DispatchData convolution_kernel_gemm_mmad8_32x3sg_128x128wg_slm_int8::SetDefault(const convolution_params& arg, int) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

        const auto& input = arg.inputs[0];
        const auto& output = arg.output;

        const auto m = input.X().v * input.Y().v *input.Batch().v;
        const auto k = input.Feature().v;
        const auto n = output.Feature().v * output.Batch().v;

        // Sub-group size used by "convolution_1x1_gemm_MMAD" kernel.
        constexpr size_t sub_group_size = 8;

        const auto of_maps = arg.output.Feature().v;
        const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);

        runInfo.effiency = FORCE_PRIORITY_1;

        runInfo.gws0 = n / (SG_TILE_N / SG_SIZE);
        runInfo.gws1 = m / SG_TILE_M;
        runInfo.gws2 = 1;

        runInfo.lws0 = SG_SIZE * WG_TILE_N / SG_TILE_N;
        runInfo.lws1 = WG_TILE_M / SG_TILE_M;
        runInfo.lws2 = 1;

        return runInfo;
    }

    JitConstants convolution_kernel_gemm_mmad8_32x3sg_128x128wg_slm_int8::GetJitConstants(const convolution_params& params, const DispatchData& runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", 8));

        const auto& input = params.inputs[0];
        const auto& output = params.output;

        const auto m = input.X().v * input.Y().v *input.Batch().v;
        const auto k = input.Feature().v;
        const auto n = output.Feature().v * output.Batch().v;

        // pitch for special block format used in this kernel
        const size_t ifm_32_aligned = Align(params.weights.IFM().v, 32);
        const size_t filter_ofm_block_pitch = (ifm_32_aligned / 32) * params.weights.X().v * params.weights.Y().v * 4 * 8 * 8;
        jit.AddConstant(MakeJitConstant("FILTER_OFM_BLOCK_PITCH", filter_ofm_block_pitch));
        jit.AddConstant(MakeJitConstant("M", m));
        jit.AddConstant(MakeJitConstant("K", k));
        jit.AddConstant(MakeJitConstant("N", n));

        return jit;
    }

    KernelsData convolution_kernel_gemm_mmad8_32x3sg_128x128wg_slm_int8::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}