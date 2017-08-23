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

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(convolution_bfyx_1x1)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    uint split_idx)
{
    const uint xy = get_group_id(0) * 16 + get_sub_group_local_id();
    const uint x = xy % OUTPUT_SIZE_X;
    const uint y = xy / OUTPUT_SIZE_Y;
    const uint f = get_global_id(1);
    const uint b = get_global_id(2);
    const uint group_f = get_group_id(1) * 16;

    UNIT_TYPE dotProd = UNIT_VAL_ZERO;
#if BIAS_TERM
    #if   BIAS_PER_OUTPUT
        const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
    #elif BIAS_PER_OFM
        const uint bias_index = f;
    #endif
    dotProd = biases[bias_index];
#endif

    const int input_x = x;
    const int input_y = y;

#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (f / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif
    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    UNIT_TYPE out[16];
    for(uint i = 0; i < 16; i++)
    {
        out[i] = dotProd;
    }

    for (uint k = 0; k < FILTER_IFM_NUM; ++k)
    {
            uint input_idx = input_offset + xy + k*INPUT0_FEATURE_PITCH;
            UNIT_TYPE in = input[input_idx];
            uint filter_idx = filter_offset + k*FILTER_IFM_PITCH;
            UNIT_TYPE w = weights[filter_idx];
            for(uint i = 0; i < 16; i++)
            {
                UNIT_TYPE _w = intel_sub_group_shuffle(w, i);
                out[i] += in * _w;
            }
    }

    if(xy >= INPUT0_SIZE_X * INPUT0_SIZE_Y)
        return;
    
    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;

    for(uint i = 0; i < 16; i++)
    {
        const uint dst_index = GET_DATA_INDEX(OUTPUT, b, group_f+i, y, x) + out_split_offset;     
        output[dst_index] = ACTIVATION(out[i], NL_M, NL_N);
    }
}
