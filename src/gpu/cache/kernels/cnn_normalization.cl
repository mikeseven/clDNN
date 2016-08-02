/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*/

#include "include/cnn_common.cl"

inline void internal_normalize(__global const DATA_TYPE* input, int input_index, __global DATA_TYPE* output, int output_index, COUNTER_TYPE sum)
{
    // TODO BDW Compiler Bug - we are using (float) because of compiler bug that convert it into (int) instead of (half)
    DATA_TYPE base = (DATA_TYPE)NORM_K + (DATA_TYPE)((COUNTER_TYPE)ALPHA*sum * (float)NUM_ELEMENTS_DIV);
    DATA_TYPE normalization_factor = pow(base, (DATA_TYPE)-BETA);
    
    DATA_TYPE f_in = input[input_index];
    DATA_TYPE normres =  f_in*normalization_factor;
    output[output_index] = activation_function(normres, NL_M ,NL_N);
}

__kernel void normalization(__global const DATA_TYPE* input, __global DATA_TYPE* output)
{
    const unsigned int x                = get_global_id(0);
    const unsigned int y                = get_global_id(1);
    const unsigned int z                = get_global_id(2);
    const unsigned int input_index      = z * INPUT_SLICE_PITCH + y*INPUT_ROW_PITCH + x + INPUT_OFFSET;
    const unsigned int output_index     = z * OUT_SLICE_PITCH + y*OUT_ROW_PITCH + x + OUT_OFFSET;

    COUNTER_TYPE sum = 0.0f;

#ifdef ACROSS_MAPS

    unsigned int j_offset = input_index - ROUND_NORM_HALF_SIZE*INPUT_SLICE_PITCH;

    for(int j = 0 ; j < ROUND_NORM_SIZE ; j++)
    {
        const int z_idx = (j + z - ROUND_NORM_HALF_SIZE);
        bool zero = (z_idx < 0 || z_idx >= INPUT_DEPTH);
        DATA_TYPE val = zero ? 0.0f : input[j_offset];
        sum += val*val;
        j_offset += INPUT_SLICE_PITCH;
    }
    
    internal_normalize(input, input_index, output, output_index, sum);
    
#else

    const int x_start = ((int)x - ROUND_NORM_HALF_SIZE);
    const int y_start = ((int)y - ROUND_NORM_HALF_SIZE);
    unsigned int input_offset  = z * INPUT_SLICE_PITCH + y_start*INPUT_ROW_PITCH + x_start;

    for (unsigned int j = 0; j < ROUND_NORM_SIZE ; ++j) 
    {
        for (unsigned int i = 0; i < ROUND_NORM_SIZE ; ++i) 
        {
            int input_offset_x = x_start + i;
            int input_offset_y = y_start + j;
            bool zero = false;
            zero = input_offset_x < 0 ? true : zero;
            zero = input_offset_y < 0 ? true : zero;
            zero = input_offset_x >= INPUT_WIDTH ? true : zero;
            zero = input_offset_y >= INPUT_HEIGHT ? true : zero;

            DATA_TYPE val = zero ? 0.0f : input[input_offset];
            
            sum += val*val;
            ++input_offset;
        }
        input_offset += INPUT_ROW_PITCH - ROUND_NORM_SIZE;
    }

    internal_normalize(input, input_index, output, output_index, sum);
#endif
}
