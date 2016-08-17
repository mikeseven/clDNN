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

    const std::string input_defines = R"__CC(
        #define INPUT_BATCH_NUM input_size[0]
        #define INPUT_FEATURE_NUM input_size[1]
        #define INPUT_SIZE_X input_size[2]
        #define INPUT_SIZE_Y input_size[3]
    )__CC";

    const std::string fully_connected_code_xb = R"__CC(
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
    )__CC";

    const std::string fully_connected_code_xb_bx = R"__CC(
        __global uint* input_size = get_raw(input_mem);
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const int x = get_global_id(0);

        pDst[x] = 0;
        uint outXIdx = x / INPUT_BATCH_NUM;
        uint weightBatchIdx = outXIdx * WEIGHTS_BATCH_NUM;
        for (uint i = 0; i < INPUT_SIZE_X; i++)
        {
            pDst[x] += input[i] * WEIGHTS[weightBatchIdx + i];
        }
        pDst[x] += BIASES[outXIdx];
    )__CC";

    const std::string fully_connected_code_yxfn = R"__CC(
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
    )__CC";

    const std::string fully_connected_code_xb_memory = R"__CC(
        __global uint* input_size = get_raw(input_mem);
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int x = get_global_id(0);

        pDst[x] = 0;
        uint outXIdx = x / INPUT_BATCH_NUM;
        uint inputBatchIdx = x % INPUT_BATCH_NUM;
        uint weightBatchIdx = outXIdx * WEIGHTS_BATCH_NUM;
        for (uint i = 0; i < INPUT_SIZE_X; i++)
        {
            pDst[x] += input[i * INPUT_BATCH_NUM + inputBatchIdx] * weight[weightBatchIdx + i];
        }
        pDst[x] += bias[outXIdx];
    )__CC";

    const std::string fully_connected_code_xb_bx_memory = R"__CC(
        __global uint* input_size = get_raw(input_mem);
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int x = get_global_id(0);

        pDst[x] = 0;
        uint outXIdx = x / INPUT_BATCH_NUM;
        uint weightBatchIdx = outXIdx * WEIGHTS_BATCH_NUM;
        for (uint i = 0; i < INPUT_SIZE_X; i++)
        {
            pDst[x] += input[i] * weight[weightBatchIdx + i];
        }
        pDst[x] += bias[outXIdx];
    )__CC";

    const std::string fully_connected_code_yxfn_memory = R"__CC(
        __global uint* input_size = get_raw(input_mem);
        __global uint* output_size = get_raw(dst_mem);
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* weight = (const __global float*)get_data(weights_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const uint x = get_global_id(0);
        const int batch_id = x % INPUT_BATCH_NUM;
        uint neuronIdx = x / INPUT_BATCH_NUM;

        pDst[x] = bias[neuronIdx];

        uint weight_idx = 0;
        uint weight_offset = neuronIdx * INPUT_FEATURE_NUM * INPUT_SIZE_Y * INPUT_SIZE_X;
        for(int k = 0; k < INPUT_FEATURE_NUM; k++)
            for(int j = 0; j < INPUT_SIZE_Y; j++)
                for(int i = 0; i < INPUT_SIZE_X; i++)
                {
                    pDst[x] += input[(k + INPUT_FEATURE_NUM * ( i + j * INPUT_SIZE_X)) * INPUT_BATCH_NUM + batch_id] * weight[weight_offset + weight_idx++];
                } 
    )__CC";
}