/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*/

#include "include/cnn_common.cl"

#if defined (USE_CNN_EXT_REFERENCE_KERNEL)

__kernel void softmax(__global DATA_TYPE* input, __global DATA_TYPE* output)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2) / OUT_BATCH;
    const unsigned int w = get_global_id(2) / OUT_DEPTH;

    unsigned int in_depth_offset = w*INPUT_BATCH_PITCH + z*INPUT_SLICE_PITCH + INPUT_OFFSET;
    
    DATA_TYPE max_value = input[in_depth_offset];
    for (int srcY = 0; srcY < INPUT_HEIGHT; ++srcY)
    {
        for (int srcX = 0; srcX < INPUT_WIDTH; ++srcX)
        {
            const unsigned int index = in_depth_offset + srcY*INPUT_ROW_PITCH + srcX;
            max_value = max(max_value, input[index]);
        }
    }

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    float denominator = 0.0;
    for (int srcY = 0; srcY < INPUT_HEIGHT; ++srcY)
    {
        for (int srcX = 0; srcX < INPUT_WIDTH; ++srcX)
        {
            const unsigned int index = in_depth_offset + srcY*INPUT_ROW_PITCH + srcX;
            const DATA_TYPE v = input[index];
            denominator += exp(v - max_value);
        }
    }
    
    const unsigned int input_idx  = in_depth_offset + y*INPUT_ROW_PITCH + x;
    const unsigned int output_idx = w*OUT_BATCH_PITCH + z*OUT_SLICE_PITCH + y*OUT_ROW_PITCH + x + OUT_OFFSET;
    const DATA_TYPE res = exp(input[input_idx] - max_value) / (DATA_TYPE)denominator;
    
    output[output_idx] = res;
}

#else

#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


DATA_TYPE find_max_value(__local DATA_TYPE* partial_max, const int idx, const __global DATA_TYPE* input)
{
    DATA_TYPE value = -DATA_TYPE_MAX;
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        value = max(value, input[LWS * i + idx]);
    }
    value = max(value, idx < LEFTOVERS? input[LWS * ITEMS_NUM + idx] : -DATA_TYPE_MAX);
    partial_max[idx] = value;

    barrier(CLK_LOCAL_MEM_FENCE);
    if(idx == 0)
    {
        for(int i = 1; i < LWS; i++)
        {
            partial_max[0] = max(partial_max[0], partial_max[i]);
        };
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return partial_max[0];
}

__kernel void softmax(const __global DATA_TYPE* input, __global DATA_TYPE* output)
{
    const int idx = get_local_id(0);

    __local DATA_TYPE partial_max[LWS];
    const DATA_TYPE max_value = find_max_value(partial_max, idx, input);
    
    DATA_TYPE tmp_vals[ITEMS_NUM + 1];
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        tmp_vals[i] = native_exp(input[LWS * i + idx] - max_value);
    }
    tmp_vals[ITEMS_NUM] = idx < LEFTOVERS ? native_exp(input[LWS * ITEMS_NUM + idx] - max_value) : DATA_TYPE_ZERO;

    // accumulate all values;
    __local DATA_TYPE partial_acc[LWS]; // all values accumulated;
    partial_acc[idx] = DATA_TYPE_ZERO;
    for(int i = 0; i < ITEMS_NUM + 1; i++)
    {
        partial_acc[idx] += tmp_vals[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE); // we must be sure that all threads calculated max of elements(we can remove it if simd32 and GWS <= 32
    if(idx == 0)
    {
        for(int i = 1; i < LWS; i++)
        {
            partial_acc[0] += partial_acc[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        output[LWS * i + idx] = tmp_vals[i] / partial_acc[0];
    }
    if(idx < LEFTOVERS)
        output[LWS * ITEMS_NUM + idx] = tmp_vals[ITEMS_NUM] / partial_acc[0];
}
#endif
