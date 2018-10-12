// Copyright (c) 2016-2017 Intel Corporation
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

#include "include/common.cl"

#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/mmad.cl"

#define FILTER_IFM_SLICES ((FILTER_IFM_NUM + 3) /4)
#define FILTER_SIZE_X_SLICES ((FILTER_SIZE_X + 7) / 8)

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(convolution_gpu_byx8_f4_fs_bs_yx_bsv4_fsv32)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
    __global BIAS_TYPE* biases,
    __global float* quantizations,
#if CALIBRATION_TERM
    __global float* calibrations,
#endif
    uint split_idx)
{
    const uint x = get_global_id(1) * 8;
    const uint y = get_global_id(2);

    const uint f = get_global_id(0) % OUTPUT_FEATURE_NUM;
    const uint b = get_global_id(0) / OUTPUT_FEATURE_NUM;

    int8 dotProd = 0;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;

    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    for (uint k = 0; k < FILTER_IFM_SLICES; ++k)
    {
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;

            for(uint i = 0; i < FILTER_SIZE_X_SLICES; i++)
            {
                uint input_idx = GET_DATA_BYX8_F4_INDEX(INPUT0, b, k * 4, input_offset_y, input_x + i * 8);
                int2 _input_data_01 = as_int2(intel_sub_group_block_read2((__global uint*)(input + input_idx)));
                int  _input_data_2  = as_int(intel_sub_group_block_read((__global uint*)(input + input_idx + 8 * 8)));

                int8 act_reg; // activations for MMAD
                act_reg[0] = _input_data_01[0];
                act_reg[1] = intel_sub_group_shuffle_down(_input_data_01[0], _input_data_01[1], STRIDE_SIZE_X * 1);
                act_reg[2] = intel_sub_group_shuffle_down(_input_data_01[0], _input_data_01[1], STRIDE_SIZE_X * 2);
                act_reg[3] = intel_sub_group_shuffle_down(_input_data_01[0], _input_data_01[1], STRIDE_SIZE_X * 3);
                act_reg[4] = _input_data_01[1];
                act_reg[5] = intel_sub_group_shuffle_down(_input_data_01[1], _input_data_2, STRIDE_SIZE_X * 1);
                act_reg[6] = intel_sub_group_shuffle_down(_input_data_01[1], _input_data_2, STRIDE_SIZE_X * 2);
                act_reg[7] = intel_sub_group_shuffle_down(_input_data_01[1], _input_data_2, STRIDE_SIZE_X * 3);

                uint filter_idx = GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(FILTER, f, k * 4, j, i * 8);
                int8 _w = as_int8(intel_sub_group_block_read8((__global uint*)(weights + filter_idx)));
                // MMAD on 8x 4 input channels elements for 8x outputs in WI
                dotProd = MMAD_8x8(act_reg, _w, dotProd);
            }
        }
    }

#if BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
#elif BIAS_PER_OFM
    const uint bias_index = f;
#endif

for(uint i = 0; i < 8; i++)
{
#if CALIBRATION_TERM
    dotProd[i] = (UNIT_TYPE)round(((float)dotProd[i] * quantizations[f] * I_QF + biases[bias_index]) * calibrations[f]);
#else  // CALIBRATION_TERM
    dotProd[i] = (UNIT_TYPE)round(((float)dotProd[i] * quantizations[f] * I_QF + biases[bias_index]) * O_QF);
#endif // CALIBRATION_TERM

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f, y, x+ i) + out_split_offset;

    output[dst_index] = ACTIVATION(convert_char(dotProd[i]), NL_M, NL_N);

}
}

#undef FILTER_SIZE_X_SLICES
#undef FILTER_IFM_SLICES