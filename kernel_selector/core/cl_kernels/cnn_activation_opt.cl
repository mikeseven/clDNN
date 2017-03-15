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

#if (NUM_ROWS_WI == 1) && (NUM_COLS_WI == 4)

__kernel void activation(__global DATA_TYPE* input, __global DATA_TYPE* output)
{
    const unsigned int x = get_global_id(0) * NUM_COLS_WI;
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2) / OUT_BATCH;
    const unsigned int w = get_global_id(2) / OUT_DEPTH;

    unsigned int input_offset = x  + y * INPUT_ROW_PITCH + INPUT_OFFSET; 
    unsigned int out_offset = x  + y * OUT_ROW_PITCH + OUT_OFFSET; 

    CAT(DATA_TYPE, 4) v = ((__global CAT(DATA_TYPE,4)*) (input + input_offset))[0];
    int m = NL_M;
    int n = NL_N;

    v = CAT(CAT(activation_function_,DATA_TYPE),4)(v, NL_M, NL_N);

#if (INPUT_WIDTH_MOD_COLS_WI == 0)
    *((__global CAT(DATA_TYPE,4)*)(output + out_offset)) = v;
#else
    if ((x + NUM_COLS_WI) < INPUT_WIDTH)
    {
        *((__global CAT(DATA_TYPE,4)*)(output + out_offset)) = v;
    }
    else
    {
        #if (INPUT_WIDTH_MOD_COLS_WI == 1)
            output[out_offset] = v.x;
        #elif (INPUT_WIDTH_MOD_COLS_WI == 2)
            ((__global CAT(DATA_TYPE,INPUT_WIDTH_MOD_COLS_WI)*)(output + out_offset))[0] = v.xy;
        #else // (INPUT_WIDTH_MOD_COLS_WI == 3)
            ((__global CAT(DATA_TYPE,INPUT_WIDTH_MOD_COLS_WI)*)(output + out_offset))[0] = v.xyz;
        #endif
    }
#endif
}

#endif