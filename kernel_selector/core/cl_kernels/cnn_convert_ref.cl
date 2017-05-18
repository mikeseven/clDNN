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

#if   defined(CONVERT_TYPE_U8)
#define CONVERT_TYPE unsigned char
#elif defined(CONVERT_TYPE_U16)
#define CONVERT_TYPE unsigned short
#elif defined(CONVERT_TYPE_U32)
#define CONVERT_TYPE unsigned int
#elif defined(CONVERT_TYPE_S8)
#define CONVERT_TYPE char
#elif defined(CONVERT_TYPE_S16)
#define CONVERT_TYPE short
#elif defined(CONVERT_TYPE_S32)
#define CONVERT_TYPE int
#elif defined(CONVERT_TYPE_F16)
#define CONVERT_TYPE half
#elif defined(CONVERT_TYPE_F32)
#define CONVERT_TYPE float
#endif

__kernel void convert(
    __global CONVERT_TYPE* input,
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

    output[dst_index] = activation_function((DATA_TYPE)input[src_index], NL_M, NL_N);
}
