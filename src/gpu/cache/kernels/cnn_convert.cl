/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*/

#include "include/cnn_common.cl"

#if   defined(COVERT_TYPE_U8)
#define COVERT_TYPE unsigned char
#elif defined(COVERT_TYPE_U16)
#define COVERT_TYPE unsigned short
#elif defined(COVERT_TYPE_U32)
#define COVERT_TYPE unsigned int
#elif defined(COVERT_TYPE_S8)
#define COVERT_TYPE char
#elif defined(COVERT_TYPE_S16)
#define COVERT_TYPE short
#elif defined(COVERT_TYPE_S32)
#define COVERT_TYPE int
#elif defined(COVERT_TYPE_F16)
#define COVERT_TYPE half
#elif defined(COVERT_TYPE_F32)
#define COVERT_TYPE float
#endif

__kernel void convert(
    __global COVERT_TYPE* input,
    __global DATA_TYPE* output)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
#if OUT_BATCH == 1
    const unsigned z = get_global_id(2);
    const unsigned w = 0;
#else
    const unsigned z = get_global_id(2) % OUT_DEPTH;
    const unsigned w = get_global_id(2) / OUT_DEPTH;
#endif

    const unsigned src_index = w*INPUT_BATCH_PITCH + z*INPUT_SLICE_PITCH + y*INPUT_ROW_PITCH + x + INPUT_OFFSET;
    const unsigned dst_index = w*OUT_BATCH_PITCH + z*OUT_SLICE_PITCH + y*OUT_ROW_PITCH + x + OUT_OFFSET;

    output[dst_index] = (DATA_TYPE)input[src_index];
}
