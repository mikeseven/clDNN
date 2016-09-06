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
        __global uint* input_size = get_raw(input_mem);
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
#ifdef RELU
        pDst[x] = max(pDst[x], 0.0f) + NEGATIVE_SLOPE * min(pDst[x], 0.0f);
#endif
    )__CC";

    const char fully_connected_code_xb_bx[] = R"__CC(
        __global uint* input_size = get_raw(input_mem);
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
#ifdef RELU
        pDst[x] = max(pDst[x], 0.0f) + NEGATIVE_SLOPE * min(pDst[x], 0.0f);
#endif
    )__CC";

    const char fully_connected_code_yxfn[] = R"__CC(
        __global uint* input_size = get_raw(input_mem);
        __global uint* output_size = get_raw(dst_mem);
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
#ifdef RELU
        pDst[neuronIdx] = max(pDst[neuronIdx], 0.0f) + NEGATIVE_SLOPE * min(pDst[neuronIdx], 0.0f);
#endif
    )__CC";

    const char fully_connected_code_xb_memory[] = R"__CC(
        __global uint* input_size = get_raw(input_mem);
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
#ifdef RELU
        pDst[x] = max(pDst[x], 0.0f) + NEGATIVE_SLOPE * min(pDst[x], 0.0f);
#endif
    )__CC";

    const char fully_connected_code_xb_bx_memory[] = R"__CC(
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int x = get_global_id(0);
        const uint batch_id = x % INPUT_BATCH_NUM;

        uint outXIdx = x / INPUT_BATCH_NUM;
        uint weight_offset = outXIdx * WEIGHTS_BATCH_NUM;
        float result = bias[outXIdx];
        for (uint i = 0; i < INPUT_SIZE_X; i++)
        {
            result += input[i * INPUT_BATCH_NUM + batch_id] * weight[weight_offset++];
        }
#ifdef RELU
        pDst[x] = max(result, 0.0f) + NEGATIVE_SLOPE * min(result, 0.0f);
#else
        pDst[x] = result;
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
#ifdef RELU
        pDst[x] = max(result, 0.0f) + NEGATIVE_SLOPE * min(result, 0.0f);
#else
        pDst[x] = result;
#endif
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
#ifdef RELU
        pDst[x] = max(result, 0.0f) + NEGATIVE_SLOPE * min(result, 0.0f);
#else
        pDst[x] = result;
#endif
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

#ifdef RELU
        pDst[x] = max(result, 0.0f) + NEGATIVE_SLOPE * min(result, 0.0f);
#else
        pDst[x] = result;
#endif
    )__CC";
}