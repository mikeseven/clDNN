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


#include "include/include_all.cl"



// tempGEMM = [ 1, direction, batch, 4 * hidden_size ]
// cell     = [ 1, direction, batch,     hidden_size ] optional
// output   = [ 2, direction, batch,     hidden_size ] output
KERNEL(lstm_elt)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if CELL_TERM
    ,const __global OUTPUT_TYPE* cell
#endif
    )
{
    const uint x = get_global_id(0);
    const uint b = get_global_id(1);

    ACCUMULATOR_TYPE it = input[GET_DATA_INDEX(INPUT0, 0, 0, b, x + GEMM_OFFSET_I)];
    ACCUMULATOR_TYPE ot = input[GET_DATA_INDEX(INPUT0, 0, 0, b, x + GEMM_OFFSET_O)]; // pass constant offsets here
    ACCUMULATOR_TYPE zt = input[GET_DATA_INDEX(INPUT0, 0, 0, b, x + GEMM_OFFSET_Z)];

    ACCUMULATOR_TYPE val = ACTIVATION_LOGISTIC(it) * ACTIVATION_HYPERBOLIC_TAN(zt);

#if CELL_TERM
    ACCUMULATOR_TYPE ft = input[GET_DATA_INDEX(INPUT0, 0, 0, b, x + GEMM_OFFSET_F)];
    val += cell[GET_DATA_INDEX(CELL, 0, 0, b, x)] * ACTIVATION_LOGISTIC(ft);
#endif

    output[GET_DATA_INDEX(OUTPUT, 0, 0, b, x)] = ACTIVATION_HYPERBOLIC_TAN(val) * ACTIVATION_LOGISTIC(ot);
    output[GET_DATA_INDEX(OUTPUT, 1, 0, b, x)] = (OUTPUT_TYPE)val;
}