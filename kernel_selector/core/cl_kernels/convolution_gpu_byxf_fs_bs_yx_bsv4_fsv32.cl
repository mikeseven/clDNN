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

#define OBS 8
__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(convolution)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if QUANTIZATION_TERM
    __global float* quantizations,
#endif
#if CALIBRATION_TERM
    __global float* calibrations,
#endif
    uint split_idx)
{
    const uint f = get_global_id(0) % OUTPUT_FEATURE_NUM;
    const uint b = get_global_id(0) / OUTPUT_FEATURE_NUM;

    const uint x = get_global_id(1) * OBS;
    const uint y = get_global_id(2);

    int dotProd[OBS] = { 0 };

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET;

    for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
    {
        const int input_offset_y = input_y + j;
        for (uint i = 0; i < FILTER_SIZE_X ; ++i)
        {
            const int input_offset_x = input_x + i;
            uint input_idx = input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH;
            uint filter_idx = filter_offset + j*FILTER_Y_PITCH + i*FILTER_X_PITCH;
            for (uint k = 0; k < FILTER_IFM_NUM; ++k)
            {
                for(uint r = 0; r < OBS; r++)
                {
                    dotProd[r] += (int)input[input_idx + INPUT0_X_PITCH * STRIDE_SIZE_X * r] * (int)weights[filter_idx];
                }
                input_idx += INPUT0_FEATURE_PITCH;
                filter_idx += FILTER_IFM_PITCH;
            }
        }
    }

for(uint r = 0; r < OBS; r++)
{
#if BIAS_TERM
    const uint bias_index = f;
#if CALIBRATION_TERM

    dotProd[r] = (UNIT_TYPE)round(((float)dotProd[r] * quantizations[f] * I_QF + biases[bias_index]) * calibrations[f]);
#else  // CALIBRATION_TERM
    dotProd[r] = (UNIT_TYPE)round(((float)dotProd[r] * quantizations[f] * I_QF + biases[bias_index]) * O_QF);
#endif // CALIBRATION_TERM
#endif

    const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f, y, x + r);
    output[dst_index] = ACTIVATION(convert_char(dotProd[r]), NL_M, NL_N);
}

}
