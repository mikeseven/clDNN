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

#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/mmad.cl"

KERNEL(convolution_gpu_byx8_f4_fs_bs_yx_bsv4_fsv32)(
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
    const uint x = get_global_id(1);
    const uint y = get_global_id(2);
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(0);
    const uint b = 0;
#else
    const uint f = get_global_id(0) % OUTPUT_FEATURE_NUM;
    const uint b = get_global_id(0) / OUTPUT_FEATURE_NUM;
#endif

#if QUANTIZATION_TERM
    int dotProd = 0;
#else
    UNIT_TYPE dotProd = UNIT_VAL_ZERO;
#endif
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (f / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif
    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    for (uint k = 0; k < FILTER_IFM_NUM; ++k)
    {
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;
            for (uint i = 0; i < FILTER_SIZE_X ; ++i)
            {
                const int input_offset_x = input_x + i * DILATION_SIZE_X;
         
                uint input_idx = GET_DATA_BYX8_F4_INDEX(INPUT0, b, k, input_offset_y, input_offset_x);
                uint filter_idx = GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(FILTER, f, k, j, i);
                dotProd += (int)input[input_idx] * (int)weights[filter_idx];
            }
        }
    }

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
#elif BIAS_PER_OFM
    const uint bias_index = f;
#endif
#if QUANTIZATION_TERM
#if CALIBRATION_TERM

    dotProd = (UNIT_TYPE)round(((float)dotProd * quantizations[f] * I_QF + biases[bias_index]) * calibrations[f]);
#else  // CALIBRATION_TERM
    dotProd = (UNIT_TYPE)round(((float)dotProd * quantizations[f] * I_QF + biases[bias_index]) * O_QF);
#endif // CALIBRATION_TERM
#else  // QUANTIZATION_TERM
    dotProd += (UNIT_TYPE)biases[bias_index];
#endif // QUANTIZATION_TERM
#endif

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f, y, x) + out_split_offset;

#if QUANTIZATION_TERM
    output[dst_index] = ACTIVATION(convert_char(dotProd), NL_M, NL_N);
#else
    output[dst_index] = ACTIVATION(dotProd, NL_M, NL_N);
#endif   
    
}