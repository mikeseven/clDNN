/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*/

#include "include/cnn_common.cl"

#if   defined REORDER_MODE_XYZW
inline unsigned int get_soruce_index(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
#elif defined REORDER_MODE_XYWZ
inline unsigned int get_soruce_index(unsigned int x, unsigned int y, unsigned int w, unsigned int z)
#elif defined REORDER_MODE_XWYZ
inline unsigned int get_soruce_index(unsigned int x, unsigned int w, unsigned int y, unsigned int z)
#elif defined REORDER_MODE_WXYZ
inline unsigned int get_soruce_index(unsigned int w, unsigned int x, unsigned int y, unsigned int z)
#elif defined REORDER_MODE_XZYW
inline unsigned int get_soruce_index(unsigned int x, unsigned int z, unsigned int y, unsigned int w)
#elif defined REORDER_MODE_ZXYW
inline unsigned int get_soruce_index(unsigned int z, unsigned int x, unsigned int y, unsigned int w)
#elif defined REORDER_MODE_YXZW
inline unsigned int get_soruce_index(unsigned int y, unsigned int x, unsigned int z, unsigned int w)
#endif
{ 
   return OUT_OFFSET + w*OUT_BATCH_PITCH + z*OUT_SLICE_PITCH + y*OUT_ROW_PITCH + x;
}

__kernel void reorder(
    __global DATA_TYPE* input,
    __global DATA_TYPE* output)
{
    const unsigned int y = get_global_id(0);
    const unsigned int z = get_global_id(1);
    const unsigned int w = get_global_id(2);
    
    const unsigned int src_index = INPUT_OFFSET + w*INPUT_BATCH_PITCH + z*INPUT_SLICE_PITCH + y*INPUT_ROW_PITCH;
    const unsigned int dst_index = 

    for (unsigned int x = 0 ; x < INPUT_WIDTH; x++)
    {
         output[get_soruce_index(x, y, z, w)] = input[src_index + x]
    }
}

