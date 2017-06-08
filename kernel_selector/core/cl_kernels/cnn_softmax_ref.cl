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



#include "include/cnn_common.cl"

KERNEL(softmax)(__global DATA_TYPE* input, __global DATA_TYPE* output)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
    const unsigned w = get_global_id(2);

    const unsigned int in_depth_offset = w*INPUT_BATCH_PITCH + y*INPUT_Y_PITCH + x + INPUT_OFFSET;
    const unsigned int out_depth_offset = w*OUT_BATCH_PITCH + y*OUT_Y_PITCH + x + OUT_OFFSET;
    
    DATA_TYPE max_value = input[in_depth_offset];
    for (int srcZ = 0; srcZ < INPUT_DEPTH; ++srcZ)
    {
        const unsigned int index = in_depth_offset + srcZ*INPUT_FEATURE_PITCH;
        max_value = fmax(max_value, input[index]);
    }

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    float denominator = 0.0;
    for (int srcZ = 0; srcZ < INPUT_DEPTH; ++srcZ)
    {
        const unsigned int index = in_depth_offset + srcZ*INPUT_FEATURE_PITCH;
        const DATA_TYPE v = input[index];
        denominator += exp(v - max_value);
    }
    
    for (int srcZ = 0; srcZ < INPUT_DEPTH; ++srcZ)
    {
        const unsigned int input_idx  = in_depth_offset + srcZ*INPUT_FEATURE_PITCH;
        const unsigned int output_idx = out_depth_offset + srcZ*OUT_FEATURE_PITCH;
        const DATA_TYPE res = exp(input[input_idx] - max_value) / (DATA_TYPE)denominator;
        output[output_idx] = FUNC_CALL(activation_function)(res, NL_M, NL_N);
    }
}