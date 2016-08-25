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

#include "convolution_common_gpu.h"

namespace neural {

    const char convolution_code_yxfb[] = R"__CC(
        const __global uint* input_size = get_raw(input_mem);
        const __global uint* dst_size = get_raw(dst_mem);
        const __global float* input = (const __global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        int global_id = get_global_id(0);
        const int batch_num = dst_size[0];
        const int batch_offset = global_id % batch_num;

        const int ofm_offset = (global_id / batch_num) % (OUTPUT_FEATURE_NUM / FILTER_ARRAY_NUM);

        const int f_ofm_offset = ofm_offset * FILTER_SIZE_Y * FILTER_SIZE_X * FILTER_INPUT_FEATURE_NUM;

        const int idx = (global_id / batch_num) / FILTER_ARRAY_NUM;

        const int i_ifm_num = input_size[1];

        const int x = ((idx / FILTER_OUTPUT_FEATURE_NUM) % dst_size[2]) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
        const int y = ((idx / FILTER_OUTPUT_FEATURE_NUM) / dst_size[2] * STRIDE_SIZE_Y) + INPUT_OFFSET_SIZE_Y;

        const int split_idx = ((global_id / batch_num) / FILTER_OUTPUT_FEATURE_NUM) % FILTER_ARRAY_NUM;
        pDst[global_id] = BIAS[split_idx][ofm_offset];

        bool finish = false;
        const uint out_x = global_id % OUTPUT_SIZE_X;
        const uint out_y = (global_id % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y)) / OUTPUT_SIZE_X;

        finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
        finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

        if(!finish)
        {
            for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
            {
                const int f_ifm_offset = h * FILTER_SIZE_Y * FILTER_SIZE_X;
                for (uint i = 0; i < FILTER_SIZE_Y; i++)
                {
                    for (uint j = 0; j < FILTER_SIZE_X; j++)
                    {
                        int input_offset_x = x + j;
                        int input_offset_y = y + i;

                        bool zero = false;
                        zero = input_offset_x < 0 ? true : zero;
                        zero = input_offset_y < 0 ? true : zero;
                        zero = input_offset_x >= input_size[2] ? true : zero;
                        zero = input_offset_y >= input_size[3] ? true : zero;

                        int input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * i_ifm_num * batch_num;
                        input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * batch_num;
                        input_idx += h * batch_num;
                        input_idx += batch_offset;
                        int filter_idx = (i * FILTER_SIZE_X + j) + f_ofm_offset + f_ifm_offset;
                        pDst[global_id] += zero ? 0 : input[input_idx] * FILTER[split_idx][filter_idx];
                    }
                }
            }
        }
        
    )__CC";

    const char convolution_code_bfxy[] = R"__CC(
        const __global uint* input_size = get_raw(input_mem);
        const __global uint* dst_size = get_raw(dst_mem);
        const __global float* input = (const __global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const int global_id = get_global_id(0);

        const int output_feature_num = dst_size[1];
        const int output_feature_size = dst_size[2] * dst_size[3];
        const int output_batch_size = output_feature_num * output_feature_size;

        const int output_feature_idx = (global_id / output_feature_size ) % output_feature_num;
        const int batch_idx = global_id / output_batch_size;

        const int filter_input_feature_size = FILTER_SIZE_X * FILTER_SIZE_Y;

        const int filter_output_feature_num = FILTER_OUTPUT_FEATURE_NUM;
        const int filter_output_feature_size = FILTER_INPUT_FEATURE_NUM * filter_input_feature_size;
        const int filter_output_feature_offset = output_feature_idx * filter_output_feature_size;
    
        const int input_feature_num = input_size[1];
        const int input_feature_size = input_size[2] * input_size[3];

        const int input_batch_size = input_feature_num * input_feature_size;
        const int input_batch_offset = input_batch_size * batch_idx;

        const int input_x_offset = global_id % (input_size[2] / STRIDE_SIZE_X) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;    
        const int input_y_offset = ((global_id / (input_size[2] / STRIDE_SIZE_X)) % (input_size[3] / STRIDE_SIZE_Y)) * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;
    
        const int input_offset = input_batch_offset + input_y_offset * input_size[2] + input_x_offset;
    
        pDst[global_id] = 0;
        for(uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
        {
            const int filter_input_feature_offset = h * filter_input_feature_size;   
            const int input_feature_offset = h * input_feature_size;
            for( uint i = 0; i < FILTER_SIZE_Y; i++)
            {
                for (uint j = 0; j < FILTER_SIZE_X; j++)
                {
                    int input_idx = j + i * input_size[2] + input_offset + input_feature_offset;
                    int filter_idx = (i * FILTER_SIZE_X + j) + filter_output_feature_offset + filter_input_feature_offset;
                    pDst[global_id] += input[input_idx] * FILTER[0][filter_idx];
                }
            }
        }
        // TODO!!!! change [0] from BIAS and FILTER to something that works - [0] is for temporary compilation
        pDst[global_id] += BIAS[0][output_feature_idx];
    )__CC";

    const char convolution_code_yxfb_memory[] = R"__CC(
        const __global float* input = (const __global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* filter = (const __global float*)get_data(filter_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int batch_num = INPUT_BATCH_NUM;

        const int bifn_num = batch_num * FILTER_OUTPUT_FEATURE_NUM;
        int global_id = get_global_id(0) % bifn_num + (get_global_id(0) / bifn_num) * bifn_num * FILTER_ARRAY_NUM + split_idx * bifn_num;

        const int ofm_offset = (global_id / batch_num) % (OUTPUT_FEATURE_NUM / FILTER_ARRAY_NUM);

        float result = bias[ofm_offset];
        
        bool finish = false;
        const uint out_x = global_id % OUTPUT_SIZE_X;
        const uint out_y = (global_id % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y)) / OUTPUT_SIZE_X;

        finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
        finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

        if(!finish)
        {
            const int batch_offset = global_id % batch_num;

            const int f_ofm_offset = ofm_offset * FILTER_SIZE_Y * FILTER_SIZE_X * FILTER_INPUT_FEATURE_NUM;

            const int idx = (global_id / batch_num) / FILTER_ARRAY_NUM;

            const int x = ((idx / FILTER_OUTPUT_FEATURE_NUM) % OUTPUT_SIZE_X) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
            const int y = ((idx / FILTER_OUTPUT_FEATURE_NUM) / OUTPUT_SIZE_X * STRIDE_SIZE_Y) + INPUT_OFFSET_SIZE_Y;

            for (uint i = 0; i < FILTER_SIZE_Y; i++)
            {
                int input_offset_y = y + i;
                bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;

                if(!zero_y)
                {
                    for (uint j = 0; j < FILTER_SIZE_X; j++)
                    {
                        int input_offset_x = x + j;
                    
                        bool zero = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;
    
                        if(!zero)
                        {
                            int input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * INPUT_FEATURE_NUM * batch_num;
                            input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * batch_num;
                            input_idx += batch_offset;
                    
                            int filter_idx = (i * FILTER_SIZE_X + j) + f_ofm_offset;

                            for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
                            {
                                const int f_ifm_offset = h * FILTER_SIZE_Y * FILTER_SIZE_X;
    
                                result += input[input_idx + h * batch_num] * filter[filter_idx + f_ifm_offset];
                            }
                        }
                    } 
                }
            }
        }
        pDst[global_id] = result;
    )__CC";

    const char convolution_code_yxfb_yxoi_memory[] = R"__CC(
        const __global float* input = (const __global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* filter = (const __global float*)get_data(filter_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int batch_num = INPUT_BATCH_NUM;

        const int bifn_num = batch_num * FILTER_OUTPUT_FEATURE_NUM;
        int global_id = get_global_id(0) % bifn_num + (get_global_id(0) / bifn_num) * bifn_num * FILTER_ARRAY_NUM + split_idx * bifn_num;

        const int ofm_offset = (global_id / batch_num) % (OUTPUT_FEATURE_NUM / FILTER_ARRAY_NUM);

        float result = bias[ofm_offset];

        bool finish = false;
        const uint out_x = global_id % OUTPUT_SIZE_X;
        const uint out_y = (global_id % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y)) / OUTPUT_SIZE_X;

        finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
        finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

        if(!finish)
        {
            const int batch_offset = global_id % batch_num;

            const int idx = (global_id / batch_num) / FILTER_ARRAY_NUM;

            const int x = ((idx / FILTER_OUTPUT_FEATURE_NUM) % OUTPUT_SIZE_X) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
            const int y = ((idx / FILTER_OUTPUT_FEATURE_NUM) / OUTPUT_SIZE_X * STRIDE_SIZE_Y) + INPUT_OFFSET_SIZE_Y;

            for (uint i = 0; i < FILTER_SIZE_Y; i++)
            {
                int input_offset_y = y + i;
                bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;

                if(!zero_y)
                {
                    for (uint j = 0; j < FILTER_SIZE_X; j++)
                    {
                        int input_offset_x = x + j;
                    
                        bool zero = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;
    
                        if(!zero)
                        {
                            int input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * INPUT_FEATURE_NUM * batch_num;
                            input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * batch_num;
                            input_idx += batch_offset;
                    
                            int filter_idx = FILTER_INPUT_FEATURE_NUM * ( ofm_offset +  FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));

                            for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
                            {
                                result += input[input_idx + h * batch_num] * filter[filter_idx + h];
                            }
                        }
                    } 
                }
            }
        }
        pDst[global_id] = result;
    )__CC";
}