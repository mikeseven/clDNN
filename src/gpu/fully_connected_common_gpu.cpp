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

    const std::string fully_connected_code_yxfn = R"__CC(
        __global uint* input_size = get_raw(input_mem);
        __global float* input = (__global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const int x = get_global_id(0);

        pDst[x] = 0;
        uint neuronIdx = x % INPUT_BATCH_NUM;
        for (uint i = 0; i < INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_FEATURE_NUM; i++)
        {
            pDst[x] += input[i * INPUT_BATCH_NUM + neuronIdx] * WEIGHTS[i];
        }
        pDst[x] += BIASES[neuronIdx];
    )__CC";
}