/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*/

#include "include/cnn_common.cl"

__kernel void locally_connected(
    __global DATA_TYPE* input, 
    __global DATA_TYPE* output, 
    __global DATA_TYPE* weights, 
    __global DATA_TYPE* biases)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2);
    
    const unsigned bias_offset = z*OUT_WIDTH*OUT_HEIGHT + y*OUT_WIDTH + x;
    DATA_TYPE dotProd = biases[bias_offset];
    
    const int conv_size = KERNEL_HEIGHT * KERNEL_WIDTH;

    for (int k = 0; k < INPUT_DEPTH; ++k)
    {
        for (int j = 0; j < KERNEL_HEIGHT; ++j)
        {
            for (int i = 0; i < KERNEL_WIDTH; ++i)
            {
                const int src_x = x * STRIDE_X + i - INPUT_PADDING_X;
                const int imgY = y * STRIDE_Y + j - INPUT_PADDING_Y;
                const int src_y = INPUT_HEIGHT * k + imgY;

                if (src_x < 0 || src_x >= INPUT_WIDTH || imgY < 0 || imgY >= INPUT_HEIGHT)
                    continue;

                const int conv_idx = z * OUT_HEIGHT * OUT_WIDTH * INPUT_DEPTH * conv_size
                    + y * OUT_WIDTH * INPUT_DEPTH * conv_size
                    + x * INPUT_DEPTH * conv_size
                    + k * conv_size + j * KERNEL_WIDTH + i;

                const int input_idx = src_y*INPUT_ROW_PITCH + src_x + INPUT_OFFSET;
                
                const DATA_TYPE w = weights[conv_idx];
                const DATA_TYPE v = input[input_idx];
                dotProd += w *v;
            }
        } 
    }
    
    unsigned int output_idx = z*OUT_SLICE_PITCH + y*OUT_ROW_PITCH + x + OUT_OFFSET;
    output[output_idx] = activation_function(dotProd, NL_M, NL_N);
}
