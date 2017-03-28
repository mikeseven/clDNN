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
#elif defined REORDER_MODE_ZYXW
inline unsigned int get_soruce_index(unsigned int z, unsigned int y, unsigned int x, unsigned int w)
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

    for (unsigned int x = 0 ; x < INPUT_WIDTH; x++)
    {
         output[get_soruce_index(x, y, z, w)] = input[src_index + x];
    }
}

