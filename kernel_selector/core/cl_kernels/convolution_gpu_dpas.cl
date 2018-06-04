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
#include "include/activation_functions.cl"
#include "include/data_types.cl"
#include "include/fetch.cl"

inline int FUNC(dp4a_SW)(char4 input, char4 weight, int acc)
{
	acc += (input[0] * weight[0]);
	acc += (input[1] * weight[1]);
	acc += (input[2] * weight[2]);
	acc += (input[3] * weight[3]);
	return acc;
}

inline int FUNC(dp4a_s8)(int8 A_scalars, int8 B_vectors, int acc)
{
	acc = FUNC_CALL(dp4a_SW)(as_char4(A_scalars[0]), as_char4(B_vectors[0]), acc);
	acc = FUNC_CALL(dp4a_SW)(as_char4(A_scalars[1]), as_char4(B_vectors[1]), acc);
	acc = FUNC_CALL(dp4a_SW)(as_char4(A_scalars[2]), as_char4(B_vectors[2]), acc);
	acc = FUNC_CALL(dp4a_SW)(as_char4(A_scalars[3]), as_char4(B_vectors[3]), acc);
	acc = FUNC_CALL(dp4a_SW)(as_char4(A_scalars[4]), as_char4(B_vectors[4]), acc);
	acc = FUNC_CALL(dp4a_SW)(as_char4(A_scalars[5]), as_char4(B_vectors[5]), acc);
	acc = FUNC_CALL(dp4a_SW)(as_char4(A_scalars[6]), as_char4(B_vectors[6]), acc);
	acc = FUNC_CALL(dp4a_SW)(as_char4(A_scalars[7]), as_char4(B_vectors[7]), acc);

	return acc;
}

#if DPAS_SUPPORTED == 1
// here declare compiler DPAS intrinsic
#else
#define DPAS(A, B, C) FUNC_CALL(dp4a_s8)(A, B, C)
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(convolution_DPAS)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global int* weights, 
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
	const uint filter_ifm_dpas_num = ((FILTER_IFM_NUM + 31) / 32);
	const uint filter_ifm_aligned = ((FILTER_IFM_NUM + 31) / 32) * 32;

    const uint filter_ofm_dpas_num = ((FILTER_OFM_NUM + 7) / 8);
    const uint filter_ofm_aligned = ((FILTER_OFM_NUM + 7) / 8) * 8;

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(2);
    const uint b = 0;
#else
    const uint f = get_global_id(2) % filter_ofm_aligned;
    const uint b = get_global_id(2) / filter_ofm_aligned;
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
// calculate alignment of input features
    const uint f32_aligned = ((FILTER_IFM_NUM + 31)/32) * 32;
    const uint filter_ofm_pitch = (f32_aligned/32) * FILTER_SIZE_X * FILTER_SIZE_Y * 4 * 8 * 8;
// end of calculations

    const uint filter_offset = (get_group_id(2) % filter_ofm_dpas_num) * filter_ofm_pitch;//f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    for (uint k = 0; k < filter_ifm_dpas_num; ++k)
    {
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;
            const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

            if(!zero_y)
            {
                for (uint i = 0; i < FILTER_SIZE_X ; ++i)
                {
                    const int input_offset_x = input_x + i * DILATION_SIZE_X;
                    const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                    if(!zero_x)
                    {
                        uint input_idx = input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH + k*32;//8 not 32 because we load ints that store 4x char
                        uint filter_idx = filter_offset + k*FILTER_Y_PITCH * FILTER_SIZE_Y + j*FILTER_Y_PITCH + i*FILTER_X_PITCH;
						filter_idx /= 4; // divide by 4 because we load in a packs of 4x char 
#if QUANTIZATION_TERM
						int input_data = as_int(intel_sub_group_block_read((const __global uint*)(input + input_idx)));
						int8 activations;  //activations of all lanes
						activations.s0 = sub_group_broadcast(input_data, 0); 
                        activations.s1 = sub_group_broadcast(input_data, 1); 
                        activations.s2 = sub_group_broadcast(input_data, 2); 
                        activations.s3 = sub_group_broadcast(input_data, 3); 
                        activations.s4 = sub_group_broadcast(input_data, 4); 
                        activations.s5 = sub_group_broadcast(input_data, 5); 
                        activations.s6 = sub_group_broadcast(input_data, 6); 
                        activations.s7 = sub_group_broadcast(input_data, 7); 

						int8 weights_data = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + filter_idx)));

						dotProd = DPAS(activations, weights_data, dotProd);

/*#if FILTER_IFM_NUM == 48 && FILTER_OFM_NUM == 192 && FILTER_SIZE_X == 3 && INPUT0_SIZE_X ==14
if(x==0 && y==0 && f==0 && i==1 && j==1)
{
	uchar4 ddd = as_uchar4(input_data);
    uchar4 ddd2 = as_uchar4(activations.s1);
    uchar4 ddd3 = as_uchar4(activations.s2);
	char4 www = as_char4(weights_data[0]);
    char4 www2 = as_char4(weights_data[1]);
	int test_mul = ddd[0] * www[0];
	printf("input int: %d, as uchar %u %u %u %u : %u %u %u %u : %u %u %u %u input idx: %d\n", input_data,
     ddd[0], ddd[1], ddd[2], ddd[3],
     ddd2[0], ddd2[1], ddd2[2], ddd2[3],
     ddd3[0], ddd3[1], ddd3[2], ddd3[3],
     input_idx);
	printf("weights int: %d, as char %d %d %d %d : %d %d %d %d weights_idx: %u\n", weights_data[0], www[0], www[1], www[2], www[3], www2[0], www2[1], www2[2], www2[3], filter_idx);
	printf("dotProd: %d test_mul: %d \n", (int)dotProd, (int)test_mul);
}
#endif*/

#else
                        dotProd += input[input_idx] * weights[filter_idx];
#endif                     
                    }
                }
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

/*#if FILTER_IFM_NUM == 48 && FILTER_OFM_NUM == 192 && FILTER_SIZE_X == 3 && INPUT0_SIZE_X ==14
if(x==0 && y==0 && f==0)
{
	printf("Quant F: %f IQF: %f bias: %f calibrations: %f dotProd: %d\n", quantizations[f], (float)I_QF, (float)biases[bias_index], calibrations[f], dotProd);
}
#endif*/

    dotProd = (UNIT_TYPE)round(((float)dotProd * quantizations[f] * I_QF + biases[bias_index]) * calibrations[f]);
#else  // CALIBRATION_TERM
    dotProd = (UNIT_TYPE)round(((float)dotProd * quantizations[f] * I_QF + biases[bias_index]) * O_QF);
#endif // CALIBRATION_TERM
#else  // QUANTIZATION_TERM
    dotProd += (UNIT_TYPE)biases[bias_index];
#endif // QUANTIZATION_TERM
#endif

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f, y, x) + out_split_offset;
#if QUANTIZATION_TERM
    output[dst_index] = ACTIVATION(convert_char(dotProd), NL_M, NL_N);
#else
    output[dst_index] = ACTIVATION(dotProd, NL_M, NL_N);
#endif   
    
}
