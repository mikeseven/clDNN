/*
// Copyright (c) 2018 Intel Corporation
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

#include "eltwise_kernel_fs_bs_yx_bsv4_fsv32.h"
#include "kernel_selector_utils.h" 

namespace kernel_selector {

    ParamsKey EltwiseKernel_fs_bs_yx_bsv4_fsv32::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableInputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
        k.EnableOutputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnableInt8Quantization();
        k.EnableOutputCalibration();
        return k;
    }

    JitConstants EltwiseKernel_fs_bs_yx_bsv4_fsv32::GetJitConstants(const eltwise_params& params) const
    {
        auto jit = EltwiseKernelBase::GetJitConstants(params);

        const size_t in_x_pitch = 32 * 4;
        const size_t in_y_pitch = 32 * 4 * params.inputs[0].X().LogicalDimPadded();
        const size_t in_b_block_pitch = in_y_pitch * params.inputs[0].Y().LogicalDimPadded();
        const size_t in_f_block_pitch = in_b_block_pitch * ((params.inputs[0].Batch().v + 3) / 4);
        const size_t in_offset = in_x_pitch * params.inputs[0].X().pad.before + in_y_pitch * params.inputs[0].Y().pad.before;

        jit.AddConstant(MakeJitConstant("IN_X_PITCH", in_x_pitch));
        jit.AddConstant(MakeJitConstant("IN_Y_PITCH", in_y_pitch));
        jit.AddConstant(MakeJitConstant("IN_B_BLOCK_PITCH", in_b_block_pitch));
        jit.AddConstant(MakeJitConstant("IN_F_BLOCK_PITCH", in_f_block_pitch));
        jit.AddConstant(MakeJitConstant("IN_OFFSET", in_offset));

        return jit;
    }

    KernelsData EltwiseKernel_fs_bs_yx_bsv4_fsv32::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}