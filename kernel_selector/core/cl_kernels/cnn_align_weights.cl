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

__kernel void align_weights(
    __global const DATA_TYPE* src,
    __global DATA_TYPE* dst
)
{
#if defined(WITH_SRC_SIZE)
    const int src_index = get_global_id(0);
    #if (INPUT_WIDTH == INPUT_ROW_PITCH)
    const int dst_index = src_index;
    #else
    const int dst_index = (src_index / INPUT_WIDTH) * INPUT_ROW_PITCH + src_index % INPUT_WIDTH;
    #endif

    dst[dst_index] = src[src_index];
#else
    const int dst_index = get_global_id(0);
    #if (INPUT_WIDTH == INPUT_ROW_PITCH)

    dst[dst_index] = src[dst_index];

    #else // #if (INPUT_WIDTH == INPUT_ROW_PITCH)

    DATA_TYPE a;
    const int x = dst_index % INPUT_ROW_PITCH;
    if (x >= INPUT_WIDTH)
    {
        a = (DATA_TYPE)(0);
    }
    else
    {
        a = src[(dst_index / INPUT_ROW_PITCH) * INPUT_WIDTH + x];
    }
    dst[dst_index] = a;

    #endif

#endif
}
