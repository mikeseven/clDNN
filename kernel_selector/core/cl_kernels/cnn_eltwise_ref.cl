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

__kernel void eltwise(
    __global DATA_TYPE* input0,
    __global DATA_TYPE* input1,
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

    const unsigned src_index0 = w*INPUT_BATCH_PITCH + z*INPUT_SLICE_PITCH + y*INPUT_ROW_PITCH + x + INPUT_OFFSET;
    const unsigned src_index1 = w*INPUT_BATCH_PITCH1 + z*INPUT_SLICE_PITCH1 + y*INPUT_ROW_PITCH1 + x + INPUT_OFFSET1;
    const unsigned dst_index = w*OUT_BATCH_PITCH + z*OUT_SLICE_PITCH + y*OUT_ROW_PITCH + x + OUT_OFFSET;

#ifdef ELTWISE_MODE_ADD
    DATA_TYPE res = input0[src_index0] + input1[src_index1];
#elif defined ELTWISE_MODE_SUB
    DATA_TYPE res = input0[src_index0] - input1[src_index1];
#elif defined ELTWISE_MODE_MUL
    DATA_TYPE res = input0[src_index0] * input1[src_index1];
#elif defined ELTWISE_MODE_DIV
    DATA_TYPE res = input0[src_index0] / input1[src_index1];
#endif

#if   defined ELTWISE_SCALAR_MODE_ADD
    res = res + (DATA_TYPE)SCALAR;
#elif defined ELTWISE_SCALAR_MODE_SUB
    res = res - (DATA_TYPE)SCALAR;
#elif defined ELTWISE_SCALAR_MODE_MUL
    res = res * (DATA_TYPE)SCALAR;
#elif defined ELTWISE_SCALAR_MODE_DIV
    res = res / (DATA_TYPE)SCALAR;
#endif
    output[dst_index] = activation_function(res, NL_M, NL_N);
}
