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

#include "include/cnn_common.cl"

__kernel void convolution(
    __global DATA_TYPE* input, 
    __global DATA_TYPE* output, 
    __global DATA_TYPE* weights, 
#ifdef OUTPUT_BIASED
    __global DATA_TYPE* biases,
#endif
    uint split_idx)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
#if OUT_BATCH == 1
    const uint z = get_global_id(2);
    const uint w = 0;
#else
    const uint z = get_global_id(2) % OUT_DEPTH;
    const uint w = get_global_id(2) / OUT_DEPTH;
#endif

    const uint filter_size = INPUT_DEPTH * KERNEL_HEIGHT * KERNEL_WIDTH;
    
#ifdef BIAS_PER_OUTPUT
    const uint bias_index = z*OUT_WIDTH*OUT_HEIGHT + y*OUT_WIDTH + x;
#else
    const uint bias_index = z;
#endif

    DATA_TYPE dotProd = (DATA_TYPE)0.0f;
#ifdef OUTPUT_BIASED
    dotProd = biases[bias_index];
#endif

    int xk_start = max((int)(INPUT_PADDING_X - x * STRIDE_X), 0);
    int yk_start = max((int)(INPUT_PADDING_Y - y * STRIDE_Y), 0);
    int xk_end = min((int)(INPUT_WIDTH - x * STRIDE_X + INPUT_PADDING_X), KERNEL_WIDTH);
    int yk_end = min((int)(INPUT_HEIGHT - y * STRIDE_Y + INPUT_PADDING_Y), KERNEL_HEIGHT);

    int input_x = max((int)(x * STRIDE_X - INPUT_PADDING_X),0);
    int input_y = max((int)(y * STRIDE_Y - INPUT_PADDING_Y),0);

    const uint in_split_offset = split_idx * INPUT_SLICE_PITCH * INPUT_DEPTH;
    uint filter_offset = z * filter_size + yk_start * KERNEL_WIDTH + xk_start;
    uint input_offset = w*INPUT_BATCH_PITCH + input_y * INPUT_ROW_PITCH + input_x + INPUT_OFFSET + in_split_offset;

    int xk_steps = xk_end - xk_start;
    int yk_steps = yk_end - yk_start;
    for (uint k = 0; k < INPUT_DEPTH; ++k)
    {
        for (uint j = yk_start; j < yk_end ; ++j)
        {
            for (uint i = xk_start; i < xk_end ; ++i)
            {
                dotProd += input[input_offset] * weights[filter_offset];
                ++input_offset;
                ++filter_offset;
            }
            input_offset +=  INPUT_ROW_PITCH - xk_steps;
            filter_offset += KERNEL_WIDTH - xk_steps;
        }
        input_offset += (INPUT_SLICE_PITCH/INPUT_ROW_PITCH - yk_steps) * INPUT_ROW_PITCH;
        filter_offset += (KERNEL_HEIGHT - yk_steps) * KERNEL_WIDTH;
    }
    
    const uint out_split_offset = split_idx * OUT_SLICE_PITCH * OUT_DEPTH;
    const uint dst_index = w*OUT_BATCH_PITCH + z*OUT_SLICE_PITCH + y*OUT_ROW_PITCH + x + OUT_OFFSET + out_split_offset;
    output[dst_index] = activation_function(dotProd, NL_M, NL_N);
}
