/*
// Copyright (c) 2018 Intel Corporation
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

#include "include/include_all.cl"

#define GET_INDEX(prefix, num) \
    GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(CAT(prefix, num), d4, d3, d2, d1) 

KERNEL(eltwise_fs_bs_yx_bsv4_fsv32)(
    INPUTS_DECLS
    __global UNIT_TYPE* output
#if CALIBRATION_TERM
    , const __global float* calibrations
#endif
    )
{
    const uint d1 = get_global_id(GWS_YX) % INPUT0_SIZE_X;   // X
    const uint d2 = get_global_id(GWS_YX) / INPUT0_SIZE_X;   // Y
    const uint d3 = get_global_id(GWS_FEATURE);             // Feature
    const uint d4 = get_global_id(GWS_BATCH);               // Batch

    uint output_offset = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, d4, d3, d2, d1);

    int res;
    
    DO_ELTWISE;

#if CALIBRATION_TERM
    res = (int)round(((float)res) * calibrations[d3]);
#else  // CALIBRATION_TERM
    res = (int)round(((float)res) * O_QF);
#endif // CALIBRATION_TERM

    output[output_offset] = ACTIVATION(convert_char(res), NL_M, NL_N);
}
