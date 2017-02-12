/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*/

#include "include/cnn_common.cl"

__kernel void binary_op(
    __global DATA_TYPE* input0,
    __global DATA_TYPE* input1,
    __global DATA_TYPE* output)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2) / OUT_BATCH;
    const unsigned int w = get_global_id(2) / OUT_DEPTH;

    const unsigned src_index = w*INPUT_BATCH_PITCH + z*INPUT_SLICE_PITCH + y*INPUT_ROW_PITCH + x + INPUT_OFFSET;
    const unsigned dst_index = w*OUT_BATCH_PITCH + z*OUT_SLICE_PITCH + y*OUT_ROW_PITCH + x + OUT_OFFSET;

#ifdef BINARY_OP_MODE_ADD
    DATA_TYPE res = input0[src_index] + input1[src_index] + (DATA_TYPE)SCALAR;
#elif defined BINARY_OP_MODE_SUB
    DATA_TYPE res = input0[src_index] - input1[src_index] + (DATA_TYPE)SCALAR;
#elif defined BINARY_OP_MODE_MUL
    DATA_TYPE res = input0[src_index] * input1[src_index] + (DATA_TYPE)SCALAR;
#elif defined BINARY_OP_MODE_DIV
    DATA_TYPE res = input0[src_index] / input1[src_index] + (DATA_TYPE)SCALAR;
#endif
    output[dst_index] = activation_function(res, NL_M, NL_N);
}
