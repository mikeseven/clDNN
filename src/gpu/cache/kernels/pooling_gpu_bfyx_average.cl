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


KERNEL(pooling_gpu_bfyx_average)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    // constexpr:
    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;


    const uint linear_id_xyz = get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2));

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    const int offset_x = x * STRIDE_SIZE_X;
    const int offset_y = y * STRIDE_SIZE_Y;

    UNIT_TYPE result = UNIT_INIT_VAL_AVG;

    const int batch_and_feature_offset = get_global_id(2);
    int input_idx = batch_and_feature_offset * INPUT_SIZE_X * INPUT_SIZE_Y + offset_y * INPUT_SIZE_X + offset_x;
    for(uint j = 0; j < WINDOW_SIZE_Y; j++)
    {
        for(uint i = 0; i < WINDOW_SIZE_X; i++)
        {
            result += input[input_idx];
            input_idx++;
        }
        input_idx += (INPUT_SIZE_X - WINDOW_SIZE_X);
    }

    const uint b = batch_and_feature_offset / INPUT_FEATURE_NUM;
    const uint f = batch_and_feature_offset % INPUT_FEATURE_NUM;
    uint output_pos = (b * OUTPUT_FEATURE_NUM + f) * output_buffer_size_x * output_buffer_size_y;
    output_pos += (OUTPUT_PADDING_LOWER_SIZE_Y + y) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X + x;

    output[output_pos] = result / (UNIT_TYPE)(WINDOW_SIZE_Y * WINDOW_SIZE_X);
}