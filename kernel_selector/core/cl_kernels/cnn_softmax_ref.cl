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
    const uint other0 = get_global_id(0);
    const uint other1 = get_global_id(1);
    const uint batch  = get_global_id(2);

    const uint in_depth_offset  = batch*INPUT_BATCH_PITCH  + other1*INPUT_OTHER1_PITCH  + other0*INPUT_OTHER0_PITCH  + INPUT_OFFSET;
    const uint out_depth_offset = batch*OUTPUT_BATCH_PITCH + other1*OUTPUT_OTHER1_PITCH + other0*OUTPUT_OTHER0_PITCH + OUTPUT_OFFSET;
    
    DATA_TYPE max_value = DATA_TYPE_MIN;
    DATA_TYPE data[INPUT_CLASS_NUM];
    
    for (uint cls = 0; cls < INPUT_CLASS_NUM; ++cls)
    {
        const uint index = in_depth_offset + cls*INPUT_CLASS_PITCH;
        DATA_TYPE in = input[index];
        max_value = max(max_value, in);
        data[cls] = in;
    }

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    COUNTER_TYPE denominator = 0.0;
    for (uint cls = 0; cls < INPUT_CLASS_NUM; ++cls)
    {
        data[cls] = native_exp(data[cls] - max_value);;
        denominator += data[cls];
    }
    
    for (uint cls = 0; cls < INPUT_CLASS_NUM; ++cls)
    {
        const DATA_TYPE res = data[cls] / (DATA_TYPE)denominator;
        const uint output_idx = out_depth_offset + cls*OUTPUT_CLASS_PITCH;
        output[output_idx] = FUNC_CALL(activation_function)(res, NL_M, NL_N);
    }
}