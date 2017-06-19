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

#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) +                                                \
    (d1 % CAT(CAT(prefix, num), _SIZES)[0])*CAT(CAT(prefix, num), _PITCHES)[0] +    \
    (d2 % CAT(CAT(prefix, num), _SIZES)[1])*CAT(CAT(prefix, num), _PITCHES)[1] +    \
    (d3 % CAT(CAT(prefix, num), _SIZES)[2])*CAT(CAT(prefix, num), _PITCHES)[2] +    \
    (d4 % CAT(CAT(prefix, num), _SIZES)[3])*CAT(CAT(prefix, num), _PITCHES)[3]


KERNEL(eltwise)(
    INPUTS_DECLS
    __global UNIT_TYPE* output)
{
    const uint d1 = get_global_id(0);
    const uint d2 = get_global_id(1);
    const uint d3 = get_global_id(2) % OUTPUT_SIZES[2];
    const uint d4 = get_global_id(2) / OUTPUT_SIZES[2];

    uint output_offset = OUTPUT_OFFSET +
                         d1*OUTPUT_PITCHES[0] +
                         d2*OUTPUT_PITCHES[1] +
                         d3*OUTPUT_PITCHES[2] +
                         d4*OUTPUT_PITCHES[3];

    UNIT_TYPE res;
    
    DO_ELTWISE;
    
    output[output_offset] = FUNC_CALL(activation_function)(res, NL_M, NL_N);
}
