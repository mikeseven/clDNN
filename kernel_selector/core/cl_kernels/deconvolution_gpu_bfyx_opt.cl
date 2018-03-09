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

#include "include/include_all.cl"

#define WORK_GROUP_GROUP_SIZE 16

__attribute__((reqd_work_group_size(WORK_GROUP_GROUP_SIZE, 1, 1)))
KERNEL(deconvolution_gpu_bfyx_opt)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* filter,
#if BIAS_TERM
    const __global UNIT_TYPE* bias,
#endif
    uint split_idx)
{
    UNIT_TYPE result = UNIT_VAL_ZERO;

    const uint b_f          = get_global_id(2);
    const uint batch_offset = b_f / OUTPUT_FEATURE_NUM;
    const uint ofm_offset   = b_f % OUTPUT_FEATURE_NUM;

    const uint global_x_group    = get_group_id(0);    
    const uint global_y_group    = get_group_id(1);

    const uint local_x        = get_local_id(0);  
    const uint local_y        = get_local_id(1);  

    const uint stride_x_id = global_x_group % STRIDE_SIZE_X;
    const uint stride_y_id = global_y_group % STRIDE_SIZE_Y;

    const uint id_x = (global_x_group / STRIDE_SIZE_X) * STRIDE_SIZE_X * WORK_GROUP_GROUP_SIZE + local_x * STRIDE_SIZE_X + stride_x_id;
    
    if (id_x >= OUTPUT_SIZE_X)
        return;

    const uint id_y = (global_y_group / STRIDE_SIZE_Y) * STRIDE_SIZE_Y + local_y * STRIDE_SIZE_Y + stride_y_id;
    const int in_x = (int)id_x + PADDING_SIZE_X - (FILTER_SIZE_X - 1);
    const int in_y = (int)id_y + PADDING_SIZE_Y - (FILTER_SIZE_Y - 1);

#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (ofm_offset / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif
    const uint input_offset = INPUT0_OFFSET + batch_offset*INPUT0_BATCH_PITCH + in_split_offset;

    for (uint i = 0; i < FILTER_SIZE_Y; i++)
    {
        const int input_offset_y = in_y + i;
        const bool zero_y = (input_offset_y >= INPUT0_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0) || ((input_offset_y % STRIDE_SIZE_Y) != 0);

        if(!zero_y)
        {
            for (uint j = 0; j < FILTER_SIZE_X; j++)
            {
                const int input_offset_x = in_x + j;
                const bool zero_x = (input_offset_x >= INPUT0_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0) || ((input_offset_x % STRIDE_SIZE_X) != 0);

                if(!zero_x)
                {
                    uint fixed_input_offset_x = (uint)input_offset_x / STRIDE_SIZE_X;
                    uint fixed_input_offset_y = (uint)input_offset_y / STRIDE_SIZE_Y;
                    uint input_idx = input_offset + (uint)fixed_input_offset_x*INPUT0_X_PITCH + (uint)fixed_input_offset_y*INPUT0_Y_PITCH;

                    uint filter_idx = ofm_offset*FILTER_OFM_PITCH + (FILTER_SIZE_Y - i - 1)*FILTER_Y_PITCH + (FILTER_SIZE_X - j - 1)*FILTER_X_PITCH;

                    for (uint h = 0; h < FILTER_IFM_NUM; h++)
                    {
                        result = fma(input[input_idx], filter[filter_idx], result);
                        filter_idx += FILTER_IFM_PITCH;
                        input_idx += INPUT0_FEATURE_PITCH;
                    }
                }
            }
        }
    }
#if BIAS_TERM
    result += bias[ofm_offset];
#endif
    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
    const uint dst_index = OUTPUT_OFFSET + out_split_offset + batch_offset*OUTPUT_BATCH_PITCH + ofm_offset*OUTPUT_FEATURE_PITCH + id_y*OUTPUT_Y_PITCH + id_x*OUTPUT_X_PITCH;
    output[dst_index] = ACTIVATION(result, NL_M, NL_N);
}

#undef ACTIVATION
#undef WORK_GROUP_GROUP_SIZE