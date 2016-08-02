/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*/

#include "include/cnn_common.cl"

__kernel void reorder(
    __global DATA_TYPE* input,
    __global DATA_TYPE* output)
{
#if 0
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2);
    
    const unsigned src_index = z*INPUT_SLICE_PITCH + y*INPUT_ROW_PITCH + x;
    const unsigned dst_index = z*OUT_SLICE_PITCH + y*OUT_ROW_PITCH + x;
#endif    
}

