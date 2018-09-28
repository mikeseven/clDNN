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

#include "include/common.cl"

#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/mmad.cl"

#define FILTER_IFM_MMAD_NUM ((FILTER_IFM_NUM + 31) / 32)
#define FILTER_OFM_MMAD_NUM ((FILTER_OFM_NUM + 7) / 8)
#define FILTER_IFM_ALIGNED (FILTER_IFM_MMAD_NUM * 32)
#define FILTER_OFM_ALIGNED (FILTER_OFM_MMAD_NUM * 8)
// input data is in blocks 4batch x 32 features
// each SIMD process 4 batches and 8 output features

#define OBS 2
#define OBH 2
#define NEEDED_INPUT_X ((OBS-1) * (STRIDE_SIZE_X) + (3 - 1) + 1)
#define NEEDED_INPUT_Y ((OBH-1) * (STRIDE_SIZE_Y) + (3 - 1) + 1)
#define WEIGHTS_PER_WORKITEM 4

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(convolution_mmad_batched)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    const __global float* quantizations,
#if CALIBRATION_TERM
    const __global float* calibrations,
#endif
    uint split_idx)
{
    const uint x = get_global_id(0) * OBS;
    const uint y = get_global_id(1) * OBH;

    const uint f = ((get_group_id(2) * WEIGHTS_PER_WORKITEM * 8) + get_sub_group_local_id() ) % FILTER_OFM_ALIGNED;
    const uint b_block = (get_group_id(2) * 8 * WEIGHTS_PER_WORKITEM) / FILTER_OFM_ALIGNED;

    int4 preloaded_input[NEEDED_INPUT_X * NEEDED_INPUT_Y];
    int4 dotProd[OBS * OBH * WEIGHTS_PER_WORKITEM] = { 0 };

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint filter_offset = ((get_group_id(2) * WEIGHTS_PER_WORKITEM) % FILTER_OFM_MMAD_NUM) * FILTER_OFM_BLOCK_PITCH;
    const uint input_offset = IN_OFFSET + IN_B_BLOCK_PITCH * b_block;

    uint filter_idx = filter_offset;
    for (uint k = 0; k < FILTER_IFM_MMAD_NUM; ++k)
    {
        // preloading data
        for(int h = 0; h < NEEDED_INPUT_Y; h++)
        for(int p = 0; p < NEEDED_INPUT_X; p++)
        {
            const int input_offset_y = input_y + h;
            const int input_offset_x = input_x + p;

            uint input_idx = input_offset + input_offset_y * IN_Y_PITCH + input_offset_x * IN_X_PITCH + k * IN_F_BLOCK_PITCH;
            preloaded_input[p + h * NEEDED_INPUT_X] = as_int4(intel_sub_group_block_read4((const __global uint*)(input + input_idx)));
        }

        __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
        for (uint j = 0; j < FILTER_SIZE_Y; ++j)
        {
            __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
            for (uint i = 0; i < FILTER_SIZE_X; ++i)
            {
                int8 weights_data[WEIGHTS_PER_WORKITEM];
                for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
                {
                    weights_data[w] = as_int8(intel_sub_group_block_read8((const __global uint*) (weights + (filter_idx + w * FILTER_OFM_BLOCK_PITCH) ) ));
                }

                for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
                for(uint h = 0; h < OBH; h++)
                for(uint o = 0; o < OBS; o++)
                {
                    const uint w_idx = (o + h * OBS + w) % WEIGHTS_PER_WORKITEM;
                    const uint out_idx = o + OBS * (h + w_idx * OBH);
                    const uint preloaded_idx =o * STRIDE_SIZE_X + i + NEEDED_INPUT_X * (h * STRIDE_SIZE_Y + j);
                    dotProd[out_idx] = MMAD_4x8(preloaded_input[preloaded_idx], weights_data[w_idx], dotProd[out_idx]);

                    //const uint out_idx = o + OBS * (h + w * OBH);
                    //const uint preloaded_idx =o * STRIDE_SIZE_X + i + NEEDED_INPUT_X * (h * STRIDE_SIZE_Y + j);
                    //dotProd[out_idx] = MMAD_4x8(preloaded_input[preloaded_idx], weights_data[w], dotProd[out_idx]);
                }
                filter_idx += FILTER_X_PITCH;
            }
        }
    }

for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
for(uint h = 0; h < OBH; h++)
for(uint o = 0; o < OBS; o++)
{
    const uint out_idx = o + OBS * (h + w * OBH);
    for(uint b = 0; b < 4; b++)
    {

    #if BIAS_TERM
        const uint bias_index = f + w * 8;
    #if CALIBRATION_TERM
        dotProd[out_idx][b] = (UNIT_TYPE)round(((float)dotProd[out_idx][b] * quantizations[f + w * 8] * I_QF + biases[bias_index]) * calibrations[f + w * 8]);
    #else  // CALIBRATION_TERM
        dotProd[out_idx][b] = (UNIT_TYPE)round(((float)dotProd[out_idx][b] * quantizations[f + w * 8] * I_QF + biases[bias_index]) * O_QF);
    #endif // CALIBRATION_TERM
    #endif // BIAS_TERM

        const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b_block*4 + b, f + w * 8, y + h, x + o);
        output[dst_index] = ACTIVATION(convert_char(dotProd[out_idx][b]), NL_M, NL_N);
    }
}
}

#undef FILTER_IFM_MMAD_NUM
#undef FILTER_OFM_MMAD_NUM
#undef FILTER_IFM_ALIGNED
#undef FILTER_OFM_ALIGNED