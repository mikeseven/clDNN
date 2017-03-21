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


#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (scale_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* scale_input
#if BIAS_TERM
, __global UNIT_TYPE* bias)
#else
)
#endif
{
#if INPUT_BFYX_USED
    const uint linear_id = (get_global_id(2) % INPUT_SIZE_X) + INPUT_SIZE_X * ((get_global_id(2) / INPUT_SIZE_X) + INPUT_SIZE_Y * (get_global_id(1) + get_global_id(0) * INPUT_FEATURE_NUM));
#else
    const uint linear_id = get_global_id(0) + INPUT_BATCH_NUM * (get_global_id(1) + INPUT_FEATURE_NUM * ((get_global_id(2) % INPUT_SIZE_X) + INPUT_SIZE_X * (get_global_id(2) / INPUT_SIZE_X)));
#endif

    const uint scale_batch_id = (SCALE_BATCH_NUM == 1) ? 0 : get_global_id(0);
    const uint scale_feature_id = (SCALE_FEATURE_NUM == 1) ? 0 : get_global_id(1);
    const uint x = (SCALE_SIZE_X == 1) ? 0 : ((SCALE_SIZE_Y == 1) ? (get_global_id(2) % INPUT_SIZE_X) : (get_global_id(2) % SCALE_SIZE_X));
    const uint y = (SCALE_SIZE_Y == 1) ? 0 : ((SCALE_SIZE_X == 1) ? (get_global_id(2) / INPUT_SIZE_X) : (get_global_id(2) / SCALE_SIZE_X));
#if SCALE_BFYX_USED
    const uint scale_linear_id = x + SCALE_SIZE_X * (y + SCALE_SIZE_Y * (scale_feature_id + scale_batch_id * SCALE_FEATURE_NUM));
#else
    const uint scale_linear_id = scale_batch_id + SCALE_BATCH_NUM * (scale_feature_id + SCALE_FEATURE_NUM * (x + y * SCALE_SIZE_X));
#endif

#if BIAS_TERM
    output[linear_id] = mad(input[linear_id], scale_input[scale_linear_id], bias[scale_linear_id]);
#else
    output[linear_id] = input[linear_id] * scale_input[scale_linear_id];
#endif
}