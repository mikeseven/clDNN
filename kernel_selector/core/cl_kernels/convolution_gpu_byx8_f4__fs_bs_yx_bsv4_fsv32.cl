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

#define FILTER_IFM_SLICES ((FILTER_IFM_NUM + 3) /4)
#define FILTER_SIZE_X_SLICES ((FILTER_SIZE_X + 7) / 8)

__attribute__((intel_reqd_sub_group_size(8)))
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

    for (uint k = 0; k < FILTER_IFM_SLICES; ++k)
    {
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;

            for(uint i = 0; i < FILTER_SIZE_X_SLICES; i++)
            {
                uint input_idx = GET_DATA_BYX8_F4_INDEX(INPUT0, b, k * 4, input_offset_y, input_x + i * 8);
                int _in = as_int(intel_sub_group_block_read((__global uint*)(input + input_idx)));

                uint filter_idx = GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(FILTER, f, k * 4, j, i * 8);
                int8 _w = as_int8(intel_sub_group_block_read8((__global uint*)(weights + filter_idx)));

                int8 activations;  //activations of all lanes
                activations.s0 = sub_group_broadcast(_in, 0); 
                activations.s1 = sub_group_broadcast(_in, 1); 
                activations.s2 = sub_group_broadcast(_in, 2); 
                activations.s3 = sub_group_broadcast(_in, 3); 
                activations.s4 = sub_group_broadcast(_in, 4); 
                activations.s5 = sub_group_broadcast(_in, 5); 
                activations.s6 = sub_group_broadcast(_in, 6); 
                activations.s7 = sub_group_broadcast(_in, 7); 

                // MMAD on 8x 4 input channels elements in WI
                dotProd = MMAD_8(activations, _w, dotProd);

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

#undef FILTER_SIZE_X_SLICES
#undef FILTER_IFM_SLICES