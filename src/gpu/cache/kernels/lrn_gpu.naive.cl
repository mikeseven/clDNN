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


#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if FP16_UNIT_USED
    #define UNIT_CVT_FUNC(val) convert_half(val)
#else
    #define UNIT_CVT_FUNC(val) (val)
#endif


KERNEL (lrn_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint global_id = get_global_id(0);
    const uint element_offset = get_global_id(1) * INPUT_BATCH_NUM * INPUT_FEATURE_NUM;

    const uint linear_id = global_id + element_offset;
    UNIT_TYPE acc = UNIT_VAL_ZERO;

    int input_offset_f = global_id + HELP_INPUT_OFFSET * INPUT_BATCH_NUM;
    int input_idx = input_offset_f + element_offset;
    for (int i = 0; i < P_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT_FEATURE_NUM * INPUT_BATCH_NUM;

        UNIT_TYPE value = zero ? UNIT_VAL_ZERO : UNIT_CVT_FUNC(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * input[input_idx];
        acc = mad(value, value, acc);

        input_offset_f += INPUT_BATCH_NUM;
        input_idx += INPUT_BATCH_NUM;
    }
    acc = mad(acc, UNIT_CVT_FUNC(ALPHA_DIV_BY_SIZE), UNIT_CVT_FUNC(K));
    acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));

    output[linear_id] = acc * input[linear_id];
}


#undef UNIT_CVT_FUNC
