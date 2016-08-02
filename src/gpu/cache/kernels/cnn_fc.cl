/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*/


#include "include/cnn_common.cl"

#if defined(__fc)
__kernel void fc(
    __global DATA_TYPE* input, 
    __global DATA_TYPE* output, 
    __global DATA_TYPE* weights, 
    __global DATA_TYPE* biases)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2); 
    
    const unsigned int input_size = INPUT_WIDTH * INPUT_HEIGHT * INPUT_DEPTH;

    unsigned int output_idx = z*OUT_SLICE_PITCH + y * OUT_ROW_PITCH + x + OUT_OFFSET;
    unsigned offset = z*OUT_WIDTH * OUT_HEIGHT + y*OUT_WIDTH + x;
    COUNTER_TYPE dotProd = (COUNTER_TYPE)(biases[offset]);

    __global DATA_TYPE* processed_neuron_weights = weights + offset * input_size;
    __global DATA_TYPE* processed_input_batch =  input;
    unsigned int weight_idx =0;

    for (unsigned int plane = 0; plane < INPUT_DEPTH; ++plane)
    {
       for (unsigned int height = 0; height < INPUT_HEIGHT; ++height)
       {
           for(unsigned int width = 0; width < INPUT_WIDTH; ++width )
           {
               unsigned int input_idx = plane*INPUT_SLICE_PITCH + height*INPUT_ROW_PITCH + width + INPUT_OFFSET;

               dotProd += (COUNTER_TYPE)(processed_input_batch[input_idx] * processed_neuron_weights[weight_idx]);

               weight_idx++;
          }
       }
    }
    output[output_idx] = activation_function((DATA_TYPE)dotProd, NL_M, NL_N);
 }
 #endif

#if defined(__fc_f16)

#define WORK_GROUP_X 64
#define VEC_SIZE 4
__attribute__ ((reqd_work_group_size(WORK_GROUP_X, 1, 1)))
__kernel void fc_f16(
    __global const half4 *src_vector,
    __global half        *dst_vector,
    __global const half  *matrix,
    __global const half  *biases)
{
    local half slm[WORK_GROUP_X];
    const int x = get_local_id(0);
    const int y = get_global_id(1);
    const int local_sz = WORK_GROUP_X;
    const int oidx = (y / OUT_WIDTH) * OUT_ROW_PITCH + y % OUT_WIDTH + OUT_OFFSET;
    int w = W;
    
    #if (LAST_INPUT_SIZE_DIV_4 == 0)
    w /= VEC_SIZE;
    __global const half4 *mat_read    = (__global const half4 *) (matrix);
    const int start_offset = w*y;
    const int end_offset = start_offset + w;
    #else
    __global const half4 *mat_read    = (__global const half4 *) (matrix + w * y);
    const int start_offset = 0;
    const int end_offset = start_offset + (w + VEC_SIZE - 1) / VEC_SIZE;
    #endif

    int m_offset = start_offset + x;
    int v_offset = INPUT_OFFSET + x;
    half4 sum = (half4)(0);
    #if (LAST_INPUT_SIZE_REMAINDER == 0)
    for (; m_offset < end_offset; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
        const half4 m = mat_read[m_offset];
        const half4 v = src_vector[v_offset];
        sum = mad(m, v, sum);
    }
    #else

        #if (LAST_INPUT_SIZE_DIV_4 == 0)
        for (; m_offset < end_offset; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
            const half4 m = mat_read[m_offset];
            const half4 v = src_vector[v_offset];

            sum = mad(m, v, sum);
        }
        #else
        for (; m_offset < end_offset - WORK_GROUP_X; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
            const half4 m = mat_read[m_offset];
            const half4 v = src_vector[v_offset];

            sum = mad(m, v, sum);
        }

        if (m_offset < end_offset)
        {
            const half4 m = mat_read[m_offset];
            const half4 v = src_vector[v_offset];
            if ((x + 1) == ((LAST_INPUT_SIZE_REMAINDER + VEC_SIZE - 1) / VEC_SIZE))
            {
                #if (LAST_INPUT_SIZE_DIV_4 == 3)
                    sum.xyz += m.xyz * v.xyz;
                #elif (LAST_INPUT_SIZE_DIV_4 == 2)
                    sum.xy += m.xy * v.xy;
                #else
                    sum.x += m.x * v.x;
                #endif
            }
            else
            {
                sum = mad(m, v, sum);
            }
        }
        #endif
    #endif

    slm[x] = sum.x + sum.y + sum.z + sum.w;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction now
    for (int max_offset = WORK_GROUP_X / 2; max_offset > 0; max_offset >>= 1) {
        if (x < max_offset) slm[x] += slm[x + max_offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #if defined(OUTPUT_BIASED)
    const half bias = biases[y];
    if (x == 0) dst_vector[oidx] = activation_function(slm[0] + bias, NL_M, NL_N);
    #else
    if (x == 0) dst_vector[oidx] = activation_function(slm[0], NL_M, NL_N);
    #endif
}
#endif 


#if defined(__fc_f32)

#define WORK_GROUP_X 64
#define VEC_SIZE 4
__attribute__ ((reqd_work_group_size(WORK_GROUP_X, 1, 1)))
__kernel void fc_f32(
    __global const float4 *src_vector,
    __global float        *dst_vector,
    __global const float  *matrix,
    __global const float  *biases)
{
    local float slm[WORK_GROUP_X];
    const int x = get_local_id(0);
    const int y = get_global_id(1);
    const int local_sz = WORK_GROUP_X;
    const int oidx = (y / OUT_WIDTH) * OUT_ROW_PITCH + y % OUT_WIDTH + OUT_OFFSET;
    int w = W;
    
    #ifdef OUTPUT_BIASED
    const float bias = biases[y];
    #else
    const float bias = 0;
    #endif

    #if (LAST_INPUT_SIZE_DIV_4 == 0)
    w /= VEC_SIZE;
    __global const float4 *mat_read    = (__global const float4 *) (matrix);
    const int start_offset = w*y;
    const int end_offset = start_offset + w;
    #else
    __global const float4 *mat_read    = (__global const float4 *) (matrix + w * y);
    const int start_offset = 0;
    const int end_offset = start_offset + (w + VEC_SIZE - 1) / VEC_SIZE;
    #endif

    int m_offset = start_offset + x;
    int v_offset = INPUT_OFFSET + x;
    float4 sum = (float4)(0);
    #if (LAST_INPUT_SIZE_REMAINDER == 0)
    for (; m_offset < end_offset; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
        const float4 m = mat_read[m_offset];
        const float4 v = src_vector[v_offset];
        sum = mad(m, v, sum);
    }
    #else

        #if (LAST_INPUT_SIZE_DIV_4 == 0)
        for (; m_offset < end_offset; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
            const float4 m = mat_read[m_offset];
            const float4 v = src_vector[v_offset];

            sum = mad(m, v, sum);
        }
        #else
        for (; m_offset < end_offset - WORK_GROUP_X; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
            const float4 m = mat_read[m_offset];
            const float4 v = src_vector[v_offset];

            sum = mad(m, v, sum);
        }

        if (m_offset < end_offset)
        {
            const float4 m = mat_read[m_offset];
            const float4 v = src_vector[v_offset];
            if ((x + 1) == ((LAST_INPUT_SIZE_REMAINDER + VEC_SIZE - 1) / VEC_SIZE))
            {
                #if (LAST_INPUT_SIZE_DIV_4 == 3)
                    sum.xyz += m.xyz * v.xyz;
                #elif (LAST_INPUT_SIZE_DIV_4 == 2)
                    sum.xy += m.xy * v.xy;
                #else
                    sum.x += m.x * v.x;
                #endif
            }
            else
            {
                sum = mad(m, v, sum);
            }
        }
        #endif
    #endif

    slm[x] = sum.x + sum.y + sum.z + sum.w;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction now
    for (int max_offset = WORK_GROUP_X / 2; max_offset > 0; max_offset >>= 1) {
        if (x < max_offset) slm[x] += slm[x + max_offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x == 0) dst_vector[oidx] = activation_function(slm[0] + bias, NL_M, NL_N);
}
#endif 

