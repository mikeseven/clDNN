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

#include "fully_connected_common_gpu.h"

namespace neural {

    const char fully_connected_code_xb[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const int x = get_global_id(0);

        pDst[x] = 0;
        uint outXIdx = x / INPUT_BATCH_NUM;
        uint inputBatchIdx = x % INPUT_BATCH_NUM;
        uint weightBatchIdx = outXIdx * WEIGHTS_BATCH_NUM;
        for (uint i = 0; i < INPUT_SIZE_X; i++)
        {
            pDst[x] += input[i * INPUT_BATCH_NUM + inputBatchIdx] * WEIGHTS[weightBatchIdx + i];
        }
        pDst[x] += BIASES[outXIdx];
		ACTIVATION(pDst[x], pDst[x]);
    )__CC";

    const char fully_connected_code_xb_bx[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const int x = get_global_id(0);
        const uint batch_id = x % INPUT_BATCH_NUM;

        uint outXIdx = x / INPUT_BATCH_NUM;
        uint weightBatchIdx = outXIdx * WEIGHTS_BATCH_NUM;
        pDst[x] = BIASES[outXIdx];
        for (uint i = 0; i < INPUT_SIZE_X; i++)
        {
            pDst[x] += input[i * INPUT_BATCH_NUM + batch_id] * WEIGHTS[weightBatchIdx + i];
        }
		ACTIVATION(pDst[x], pDst[x]);
    )__CC";

    const char fully_connected_code_yxfn[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const uint x = get_global_id(0);
        const int batch_id = x % INPUT_BATCH_NUM;
        uint neuronIdx = x / INPUT_BATCH_NUM;

        pDst[x] = BIASES[neuronIdx];

        uint weight_idx = 0;
        uint weight_offset = neuronIdx * INPUT_FEATURE_NUM * INPUT_SIZE_Y * INPUT_SIZE_X;
        for(int k = 0; k < INPUT_FEATURE_NUM; k++)
            for(int j = 0; j < INPUT_SIZE_Y; j++)
                for(int i = 0; i < INPUT_SIZE_X; i++)
                {
                    pDst[neuronIdx] += input[(k + INPUT_FEATURE_NUM * ( i + j * INPUT_SIZE_X)) * INPUT_BATCH_NUM + batch_id] * WEIGHTS[weight_offset + weight_idx++];
                } 
		 ACTIVATION(pDst[neuronIdx],pDst[neuronIdx]);
    )__CC";

    const char fully_connected_code_xb_memory[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int x = get_global_id(0);
        const uint batch_id = x % INPUT_BATCH_NUM;

        uint outXIdx = x / INPUT_BATCH_NUM;
        uint weightBatchIdx = outXIdx * WEIGHTS_BATCH_NUM;
        pDst[x] = bias[outXIdx];
        for (uint i = 0; i < INPUT_SIZE_X; i++)
        {
            pDst[x] += input[i * INPUT_BATCH_NUM + batch_id] * weight[weightBatchIdx + i];
        }

		ACTIVATION(pDst[x], pDst[x]);
    )__CC";

    const char fully_connected_code_xb_bx_memory[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int x = get_global_id(0);
        const uint batch_id = x % INPUT_BATCH_NUM;

        uint outXIdx = x / INPUT_BATCH_NUM;
        uint weight_offset = outXIdx * INPUT_ELEMENTS_COUNT;
        float result = bias[outXIdx];
        for (uint i = 0; i < INPUT_ELEMENTS_COUNT; i++)
        {
            result += input[i * INPUT_BATCH_NUM + batch_id] * weight[weight_offset++];
        }
		 ACTIVATION(pDst[x], result);
    )__CC";

    const char fully_connected_code_xb_bx_b8_memory[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int x = get_global_id(0);
        const uint batch_id = x % INPUT_BATCH_NUM;

        uint outXIdx = x / INPUT_BATCH_NUM;
        uint weight_offset = outXIdx * INPUT_ELEMENTS_COUNT + batch_id;
        float result = bias[outXIdx];

        float8 _data = 0.f;
        const uint sub_group_id = get_local_id(0);
        
        for(uint _i = 0; _i < INPUT_ELEMENTS_COUNT/8; _i++)
        {
            uint i = _i * 8;
            const float weight_val = weight[weight_offset];
            const float8 _input = as_float8(intel_sub_group_block_read8((const __global uint*)input + i * INPUT_BATCH_NUM + batch_id));
            _data.s0 = fma(_input.s0, intel_sub_group_shuffle(weight_val, 0), _data.s0);
            _data.s1 = fma(_input.s1, intel_sub_group_shuffle(weight_val, 1), _data.s1);                                
            _data.s2 = fma(_input.s2, intel_sub_group_shuffle(weight_val, 2), _data.s2);
            _data.s3 = fma(_input.s3, intel_sub_group_shuffle(weight_val, 3), _data.s3);
            _data.s4 = fma(_input.s4, intel_sub_group_shuffle(weight_val, 4), _data.s4);
            _data.s5 = fma(_input.s5, intel_sub_group_shuffle(weight_val, 5), _data.s5);
            _data.s6 = fma(_input.s6, intel_sub_group_shuffle(weight_val, 6), _data.s6);
            _data.s7 = fma(_input.s7, intel_sub_group_shuffle(weight_val, 7), _data.s7);
            weight_offset += 8;
        }
        for(uint i = INPUT_ELEMENTS_COUNT - (INPUT_ELEMENTS_COUNT % 8); i < INPUT_ELEMENTS_COUNT; i++)
        {
            result += input[i * INPUT_BATCH_NUM + batch_id] * weight[weight_offset++];
        }
        result += _data.s0 + _data.s1 + _data.s2 + _data.s3 +
                  _data.s4 + _data.s5 + _data.s6 + _data.s7;

		 ACTIVATION(pDst[x], result);
    )__CC";

    const char fully_connected_code_xb_xb_b8_x8_memory[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const uint global_id = get_global_id(0);
        const int x = get_global_id(0);
        const uint batch_id = x % INPUT_BATCH_NUM;

        uint neuronIdx = (x / INPUT_BATCH_NUM) * NEURONS_PER_WORK_ITEM;

        const uint sub_group_id = get_local_id(0);
        const uint batch_num = INPUT_BATCH_NUM;

        const int out_id = (global_id / batch_num) * NEURONS_PER_WORK_ITEM * batch_num + batch_id;

        const int ofm_offset = (global_id * NEURONS_PER_WORK_ITEM) / batch_num;

        float8 _data0 = 0.f;
#if NEURONS_PER_WORK_ITEM > 8
        float8 _data1 = 0.f;
#endif

        uint weight_offset = sub_group_id + neuronIdx;

        for(uint h = 0; h < INPUT_ELEMENTS_COUNT; h++)
        {
            DOT_PRODUCT_8(_data0, input[h * batch_num + batch_id], weight[weight_offset])
#if NEURONS_PER_WORK_ITEM > 8
            DOT_PRODUCT_8(_data1, input[h * batch_num + batch_id], weight[weight_offset + 8])
#endif
            weight_offset+= WEIGHTS_BATCH_NUM;
        }


    ADD_BIAS_8(_data0, bias[neuronIdx + sub_group_id]);
#if NEURONS_PER_WORK_ITEM > 8
    ADD_BIAS_8(_data1, bias[neuronIdx + sub_group_id + 8]);
#endif
    RELU_8(_data0);
#if NEURONS_PER_WORK_ITEM > 8
    RELU_8(_data1);
#endif
 
    intel_sub_group_block_write8((__global uint*)pDst + out_id, as_uint8(_data0));
#if NEURONS_PER_WORK_ITEM > 8
    intel_sub_group_block_write8((__global uint*)pDst + out_id + 8 * batch_num, as_uint8(_data1));
#endif
    )__CC";

    const char fully_connected_code_yxfn_memory[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const uint x = get_global_id(0);
        const int batch_id = x % INPUT_BATCH_NUM;
        uint neuronIdx = x / INPUT_BATCH_NUM;

        float result = bias[neuronIdx];

        uint weight_offset = neuronIdx * INPUT_FEATURE_NUM * INPUT_SIZE_Y * INPUT_SIZE_X;
        for(int k = 0; k < INPUT_FEATURE_NUM; k++)
            for(int j = 0; j < INPUT_SIZE_Y; j++)
                for(int i = 0; i < INPUT_SIZE_X; i++)
                {
                    result += input[(k + INPUT_FEATURE_NUM * ( i + j * INPUT_SIZE_X)) * INPUT_BATCH_NUM + batch_id] * weight[weight_offset++];
                }
		 ACTIVATION(pDst[x], result);
    )__CC";

    const char fully_connected_code_yxfn_byxf_memory[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const uint x = get_global_id(0);
        const int batch_id = x % INPUT_BATCH_NUM;
        uint neuronIdx = x / INPUT_BATCH_NUM;

        float result = bias[neuronIdx];

        uint weight_offset = neuronIdx * INPUT_FEATURE_NUM * INPUT_SIZE_Y * INPUT_SIZE_X;
        for(int j = 0; j < INPUT_SIZE_Y; j++)
            for(int i = 0; i < INPUT_SIZE_X; i++)
            {    
                int input_idx = (i + j * INPUT_SIZE_X) * INPUT_FEATURE_NUM * INPUT_BATCH_NUM + batch_id;
                for(int k = 0; k < INPUT_FEATURE_NUM; k++)
                {
                    result += input[input_idx + k * INPUT_BATCH_NUM] * weight[weight_offset++];
                }
            }
		 ACTIVATION(pDst[x], result);
    )__CC";

    const char fully_connected_code_yxfn_byxf_b8_f8_memory[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const uint x = get_global_id(0);
        const int batch_id = x % INPUT_BATCH_NUM;
        uint neuronIdx = x / INPUT_BATCH_NUM;

        float result = bias[neuronIdx];

        float8 _data = 0.f;

        const uint sub_group_id = get_local_id(0);

        uint weight_offset = sub_group_id + neuronIdx * INPUT_FEATURE_NUM * INPUT_SIZE_Y * INPUT_SIZE_X;
        for(int j = 0; j < INPUT_SIZE_Y; j++)
        {
            for(int i = 0; i < INPUT_SIZE_X; i++)
            {    
                int input_idx = (i + j * INPUT_SIZE_X) * INPUT_FEATURE_NUM * INPUT_BATCH_NUM + batch_id;
                for(int k = 0; k < INPUT_FEATURE_NUM; k+=8)
                {
                    const float weight_val = weight[weight_offset];
                    const float8 _input = as_float8(intel_sub_group_block_read8((const __global uint*)input + input_idx + k * INPUT_BATCH_NUM));
                    _data.s0 = fma(_input.s0, intel_sub_group_shuffle(weight_val, 0), _data.s0);
                    _data.s1 = fma(_input.s1, intel_sub_group_shuffle(weight_val, 1), _data.s1);                                
                    _data.s2 = fma(_input.s2, intel_sub_group_shuffle(weight_val, 2), _data.s2);
                    _data.s3 = fma(_input.s3, intel_sub_group_shuffle(weight_val, 3), _data.s3);
                    _data.s4 = fma(_input.s4, intel_sub_group_shuffle(weight_val, 4), _data.s4);
                    _data.s5 = fma(_input.s5, intel_sub_group_shuffle(weight_val, 5), _data.s5);
                    _data.s6 = fma(_input.s6, intel_sub_group_shuffle(weight_val, 6), _data.s6);
                    _data.s7 = fma(_input.s7, intel_sub_group_shuffle(weight_val, 7), _data.s7);
                    weight_offset += 8;
                }
            }
        }
        result += _data.s0 + _data.s1 + _data.s2 + _data.s3 +
                  _data.s4 + _data.s5 + _data.s6 + _data.s7;

		ACTIVATION(pDst[x], result);
    )__CC";
}