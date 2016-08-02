/*
//
//                 INTEL CORPORATION PROPRIETARY INFORMATION
//    This software is supplied under the terms of a license agreement or
//    nondisclosure agreement with Intel Corporation and may not be copied
//    or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*/

#include "include/cnn_common.cl"

#if defined(USE_CNN_EXT_REFERENCE_KERNEL)
__kernel void nonlinear(__global DATA_TYPE* input, __global DATA_TYPE* output)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int offset = x  + y * INPUT_ROW_PITCH + INPUT_OFFSET; 
	
    output[y * OUT_ROW_PITCH + x + OUT_OFFSET] = activation_function(input[offset], NL_M, NL_N);
}

#elif (NUM_ROWS_WI == 1) && (NUM_COLS_WI == 4)

__kernel void nonlinear(__global DATA_TYPE* input, __global DATA_TYPE* output)
{
    const unsigned int x = get_global_id(0) * NUM_COLS_WI;
    const unsigned int y = get_global_id(1);

    unsigned int input_offset = x  + y * INPUT_ROW_PITCH + INPUT_OFFSET; 
    unsigned int out_offset = x  + y * OUT_ROW_PITCH + OUT_OFFSET; 

    CAT(DATA_TYPE, 4) v = ((__global CAT(DATA_TYPE,4)*) (input + input_offset))[0];
    int m = NL_M;
    int n = NL_N;

    v = CAT(CAT(activation_function_,DATA_TYPE),4)(v, *((DATA_TYPE*)&m), *((DATA_TYPE*)&n));

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

#else
#error "Not supported"
#endif