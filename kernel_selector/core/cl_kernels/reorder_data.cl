// Copyright (c) 2016-2017 Intel Corporation
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


#include "include/common.cl"

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint b, uint f, uint y, uint x)
{
#if   INPUT0_SIMPLE
    return GET_DATA_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_BS_F_BSV8__AF8  || \
      defined INPUT0_LAYOUT_BS_F_BSV16__AF8
    return GET_DATA_BS_FYX_BSV8_INDEX(INPUT0, b, f, y, x, SUB_GROUP_SIZE);
#else
#error - not supported
#endif
}

///////////////////////// Output Index /////////////////////////

inline uint FUNC(get_output_index)(uint b, uint f, uint y, uint x)
{
#if   OUTPUT_SIMPLE
    return GET_DATA_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_BS_F_BSV8__AF8  || \
      defined OUTPUT_LAYOUT_BS_F_BSV16__AF8
    return GET_DATA_BS_FYX_BSV8_INDEX(OUTPUT, b, f, y, x, SUB_GROUP_SIZE);
#else
#error - not supported
#endif
}

KERNEL (reorder_data)(
    const __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output
#ifdef MEAN_SUBTRACT_IN_BUFFER
    , __global MEAN_SUBTRACT_TYPE* mean_subtruct
#endif
    )
{
    const uint b = get_global_id(GWS_BATCH);
    const uint f = get_global_id(GWS_FEATURE);
#if   INPUT0_DIMS == 2
    const uint y = 0;
    const uint x = 0;
#elif INPUT0_DIMS == 4
    const uint y = ((uint)(get_global_id(GWS_YX))) / INPUT0_SIZE_X;
    const uint x = ((uint)(get_global_id(GWS_YX))) % INPUT0_SIZE_X;
#endif

    uint4 ov = FUNC_CALL(reshape_dims)(b,f,y,x, INPUT0_SIZE_Y, INPUT0_SIZE_X, OUTPUT_SIZE_Y, OUTPUT_SIZE_X, INPUT0_DIMS, OUTPUT_DIMS);
    const uint input_idx  = FUNC_CALL(get_input_index)(b, f, y, x);
    const uint output_idx = FUNC_CALL(get_output_index)(ov[0],ov[1],ov[2],ov[3]);
    CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
    
#if   defined MEAN_SUBTRACT_INSIDE_PARAMS
    res -= TO_CALC_TYPE(VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE]);
#elif defined MEAN_SUBTRACT_IN_BUFFER
    uint4 msv = FUNC_CALL(reshape_dims)(b,f,y,x, INPUT0_SIZE_Y, INPUT0_SIZE_X, MEAN_SUBTRACT_SIZE_Y, MEAN_SUBTRACT_SIZE_X, INPUT0_DIMS, MEAN_SUBTRACT_DIMS);
    res -= TO_CALC_TYPE(mean_subtruct[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[0], msv[1], msv[2], msv[3])]);
#endif

    output[output_idx] = TO_OUTPUT_TYPE(res);
}