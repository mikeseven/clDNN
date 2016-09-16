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

namespace neural 
{

	const char convolution_code_yxfb[] = R"__CC(
        const __global float* input = (const __global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        int global_id = get_global_id(0);
        const int batch_num = OUTPUT_BATCH_NUM;
        const int batch_offset = global_id % batch_num;

        const int ofm_offset = (global_id / batch_num) % (OUTPUT_FEATURE_NUM / FILTER_ARRAY_NUM);

        const int f_ofm_offset = ofm_offset * FILTER_SIZE_Y * FILTER_SIZE_X * FILTER_INPUT_FEATURE_NUM;

        const int idx = (global_id / batch_num) / FILTER_ARRAY_NUM;

        const int i_ifm_num = INPUT_FEATURE_NUM_0;

        const int x = ((idx / FILTER_OUTPUT_FEATURE_NUM) % OUTPUT_SIZE_X) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
        const int y = ((idx / FILTER_OUTPUT_FEATURE_NUM) / OUTPUT_SIZE_X * STRIDE_SIZE_Y) + INPUT_OFFSET_SIZE_Y;

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
                        zero = input_offset_x >= INPUT_SIZE_X ? true : zero;
                        zero = input_offset_y >= INPUT_SIZE_Y ? true : zero;

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
        
    ACTIVATION(pDst[global_id], pDst[global_id]);
    )__CC";

    const char convolution_code_bfxy[] = R"__CC(
        const __global float* input = (const __global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const int global_id = get_global_id(0);

        const int output_feature_num = OUTPUT_FEATURE_NUM;
        const int output_feature_size = OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
        const int output_batch_size = output_feature_num * output_feature_size;

        const int output_feature_idx = (global_id / output_feature_size ) % output_feature_num;
        const int batch_idx = global_id / output_batch_size;

        const int filter_input_feature_size = FILTER_SIZE_X * FILTER_SIZE_Y;

        const int filter_output_feature_num = FILTER_OUTPUT_FEATURE_NUM;
        const int filter_output_feature_size = FILTER_INPUT_FEATURE_NUM * filter_input_feature_size;
        const int filter_output_feature_offset = output_feature_idx * filter_output_feature_size;
    
        const int input_feature_num = INPUT_FEATURE_NUM_0;
        const int input_feature_size = INPUT_SIZE_X * INPUT_SIZE_Y;

        const int input_batch_size = input_feature_num * input_feature_size;
        const int input_batch_offset = input_batch_size * batch_idx;

        const int input_x_offset = global_id % (INPUT_SIZE_X / STRIDE_SIZE_X) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;    
        const int input_y_offset = ((global_id / (INPUT_SIZE_X / STRIDE_SIZE_X)) % (INPUT_SIZE_Y / STRIDE_SIZE_Y)) * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;
    
        const int input_offset = input_batch_offset + input_y_offset * INPUT_SIZE_X + input_x_offset;
    
        pDst[global_id] = 0;
        for(uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
        {
            const int filter_input_feature_offset = h * filter_input_feature_size;   
            const int input_feature_offset = h * input_feature_size;
            for( uint i = 0; i < FILTER_SIZE_Y; i++)
            {
                for (uint j = 0; j < FILTER_SIZE_X; j++)
                {
                    int input_idx = j + i * INPUT_SIZE_X + input_offset + input_feature_offset;
                    int filter_idx = (i * FILTER_SIZE_X + j) + filter_output_feature_offset + filter_input_feature_offset;
                    pDst[global_id] += input[input_idx] * FILTER[0][filter_idx];
                }
            }
        }
        // TODO!!!! change [0] from BIAS and FILTER to something that works - [0] is for temporary compilation
        pDst[global_id] += BIAS[0][output_feature_idx];
        ACTIVATION(pDst[global_id], pDst[global_id]);
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
		ACTIVATION(pDst[global_id], result);
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
		ACTIVATION(pDst[global_id], result);
    )__CC";

    const char convolution_code_yxfb_oyxi_memory[] = R"__CC(
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

            const int f_ofm_offset = ofm_offset * FILTER_INPUT_FEATURE_NUM * FILTER_SIZE_X * FILTER_SIZE_Y;
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
                    
                            int filter_idx = f_ofm_offset + FILTER_INPUT_FEATURE_NUM * ( i * FILTER_SIZE_X + j);

                            for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
                            {
                                result += input[input_idx + h * batch_num] * filter[filter_idx + h];
                            }
                        }
                    } 
                }
            }
        }
		ACTIVATION(pDst[global_id], result);
    )__CC";

    const char convolution_code_yxfb_yxoi_b8_memory[] = R"__CC(
#define OFM_PER_WORK_ITEM 16
#define DOT_PRODUCT_8( _result, _rowA, colB )    \
{   \
        _result.s0 = mad( _rowA, intel_sub_group_shuffle( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, intel_sub_group_shuffle( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, intel_sub_group_shuffle( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, intel_sub_group_shuffle( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, intel_sub_group_shuffle( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, intel_sub_group_shuffle( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, intel_sub_group_shuffle( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, intel_sub_group_shuffle( colB, 7 ), _result.s7 );  \
}
#define ADD_BIAS_8( _result, _biasVal) \
{ \
    _result.s0 += intel_sub_group_shuffle( _biasVal, 0 ); \
    _result.s1 += intel_sub_group_shuffle( _biasVal, 1 ); \
    _result.s2 += intel_sub_group_shuffle( _biasVal, 2 ); \
    _result.s3 += intel_sub_group_shuffle( _biasVal, 3 ); \
    _result.s4 += intel_sub_group_shuffle( _biasVal, 4 ); \
    _result.s5 += intel_sub_group_shuffle( _biasVal, 5 ); \
    _result.s6 += intel_sub_group_shuffle( _biasVal, 6 ); \
    _result.s7 += intel_sub_group_shuffle( _biasVal, 7 ); \
}

        const __global float* input = (const __global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* filter = (const __global float*)get_data(filter_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int batch_num = INPUT_BATCH_NUM;

        const uint linear_id_xy = get_global_id(1) + get_global_size(1) * get_global_id(2);
        // we're computing 8 OUTPUT_FEATURE_MAP so we must divide by 8, but we got 8 batches, so no division is needed.
        int global_id = (get_global_id(0) / batch_num) * 8 + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * (FILTER_OUTPUT_FEATURE_NUM / OFM_PER_WORK_ITEM) * batch_num; 

        const uint out_batch_id = get_local_id(0);
        const uint out_x = get_global_id(1);
        const uint out_y = get_global_id(2);

        const int out_id = (global_id / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;

        const int ofm_offset = (global_id * (OFM_PER_WORK_ITEM / batch_num)) % FILTER_OUTPUT_FEATURE_NUM;

        bool finish = false;

        finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
        finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

        const uint sub_group_id = get_local_id(0);

        float8 _data0 = 0.f;
        float8 _data1 = 0.f;

        if(!finish)
        {
            const int x = out_x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
            const int y = out_y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

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
                            input_idx += out_batch_id;
                        
                            //sub_group_id used as offset to make each workitem load different filter, and then shuffle it
                            int filter_idx = FILTER_INPUT_FEATURE_NUM * ( ofm_offset + sub_group_id +  FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));
    
                            for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
                            {
                                DOT_PRODUCT_8(_data0, input[input_idx + h * batch_num], filter[filter_idx + h])
                                DOT_PRODUCT_8(_data1, input[input_idx + h * batch_num], filter[filter_idx + h + FILTER_INPUT_FEATURE_NUM * 8])
                            }
                        }
                    } 
                }
            }
        }

        ADD_BIAS_8(_data0, bias[ofm_offset + sub_group_id]);
        ADD_BIAS_8(_data1, bias[ofm_offset + sub_group_id + 8]);

#define RELU_8(_result) \
{ \
        ACTIVATION(_result.s0, _result.s0); \
        ACTIVATION(_result.s1, _result.s1); \
        ACTIVATION(_result.s2, _result.s2); \
        ACTIVATION(_result.s3, _result.s3); \
        ACTIVATION(_result.s4, _result.s4); \
        ACTIVATION(_result.s5, _result.s5); \
        ACTIVATION(_result.s6, _result.s6); \
        ACTIVATION(_result.s7, _result.s7); \
}
        RELU_8(_data0);
        RELU_8(_data1);

        intel_sub_group_block_write8((__global uint*)pDst + out_id, as_uint8(_data0));
        intel_sub_group_block_write8((__global uint*)pDst + out_id + 8 * batch_num, as_uint8(_data1));
    )__CC";




    const char convolution_code_yxfb_yxio_b8_memory[] = R"__CC(
#define OFM_PER_WORK_ITEM 16
#define DOT_PRODUCT_8( _result, _rowA, colB )    \
{   \
        _result.s0 = mad( _rowA, intel_sub_group_shuffle( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, intel_sub_group_shuffle( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, intel_sub_group_shuffle( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, intel_sub_group_shuffle( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, intel_sub_group_shuffle( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, intel_sub_group_shuffle( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, intel_sub_group_shuffle( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, intel_sub_group_shuffle( colB, 7 ), _result.s7 );  \
}
#define ADD_BIAS_8( _result, _biasVal) \
{ \
    _result.s0 += intel_sub_group_shuffle( _biasVal, 0 ); \
    _result.s1 += intel_sub_group_shuffle( _biasVal, 1 ); \
    _result.s2 += intel_sub_group_shuffle( _biasVal, 2 ); \
    _result.s3 += intel_sub_group_shuffle( _biasVal, 3 ); \
    _result.s4 += intel_sub_group_shuffle( _biasVal, 4 ); \
    _result.s5 += intel_sub_group_shuffle( _biasVal, 5 ); \
    _result.s6 += intel_sub_group_shuffle( _biasVal, 6 ); \
    _result.s7 += intel_sub_group_shuffle( _biasVal, 7 ); \
}

        const __global float* input = (const __global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* filter = (const __global float*)get_data(filter_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int batch_num = INPUT_BATCH_NUM;

        const uint linear_id_xy = get_global_id(1) + get_global_size(1) * get_global_id(2);
        // we're computing 8 OUTPUT_FEATURE_MAP so we must divide by 8, but we got 8 batches, so no division is needed.
        int global_id = (get_global_id(0) / batch_num) * 8 + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * (FILTER_OUTPUT_FEATURE_NUM / OFM_PER_WORK_ITEM) * batch_num; 

        const uint out_batch_id = get_local_id(0);
        const uint out_x = get_global_id(1);
        const uint out_y = get_global_id(2);

        const int out_id = (global_id / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;

        const int ofm_offset = (global_id * (OFM_PER_WORK_ITEM / batch_num)) % FILTER_OUTPUT_FEATURE_NUM;

        bool finish = false;

        finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
        finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

        const uint sub_group_id = get_local_id(0);

        float8 _data0 = 0.f;
        float8 _data1 = 0.f;

        if(!finish)
        {
            const int x = out_x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
            const int y = out_y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

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
                            input_idx += out_batch_id;
                        
                            //sub_group_id used as offset to make each workitem load different filter, and then shuffle it
                            int filter_idx = ofm_offset + sub_group_id + FILTER_INPUT_FEATURE_NUM * (FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));
    
                            for (uint _h = 0; _h < FILTER_INPUT_FEATURE_NUM/8; _h++)
                            {
                                uint h = _h*8;
                                float8 _input = as_float8(intel_sub_group_block_read8((const __global uint*)input + input_idx + h * batch_num));
                                                                
                                DOT_PRODUCT_8(_data0, _input.s0, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM])
                                DOT_PRODUCT_8(_data1, _input.s0, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM + 8]) h++;
                                
                                DOT_PRODUCT_8(_data0, _input.s1, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM])
                                DOT_PRODUCT_8(_data1, _input.s1, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM + 8]) h++;

                                DOT_PRODUCT_8(_data0, _input.s2, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM])
                                DOT_PRODUCT_8(_data1, _input.s2, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM + 8]) h++;

                                DOT_PRODUCT_8(_data0, _input.s3, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM])
                                DOT_PRODUCT_8(_data1, _input.s3, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM + 8]) h++;

                                DOT_PRODUCT_8(_data0, _input.s4, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM])
                                DOT_PRODUCT_8(_data1, _input.s4, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM + 8]) h++;

                                DOT_PRODUCT_8(_data0, _input.s5, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM])
                                DOT_PRODUCT_8(_data1, _input.s5, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM + 8]) h++;

                                DOT_PRODUCT_8(_data0, _input.s6, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM])
                                DOT_PRODUCT_8(_data1, _input.s6, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM + 8]) h++;

                                DOT_PRODUCT_8(_data0, _input.s7, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM])
                                DOT_PRODUCT_8(_data1, _input.s7, filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM + 8]) h++;
                            }
                            for (uint h = FILTER_INPUT_FEATURE_NUM - (FILTER_INPUT_FEATURE_NUM % 8); h < FILTER_INPUT_FEATURE_NUM; h++)
                            {
                                DOT_PRODUCT_8(_data0, input[input_idx + h * batch_num], filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM])
                                DOT_PRODUCT_8(_data1, input[input_idx + h * batch_num], filter[filter_idx + h * FILTER_OUTPUT_FEATURE_NUM + 8])
                            }
                        }
                    } 
                }
            }
        }

        ADD_BIAS_8(_data0, bias[ofm_offset + sub_group_id]);
        ADD_BIAS_8(_data1, bias[ofm_offset + sub_group_id + 8]);

#define RELU_8(_result) \
{ \
        ACTIVATION(_result.s0, _result.s0); \
        ACTIVATION(_result.s1, _result.s1); \
        ACTIVATION(_result.s2, _result.s2); \
        ACTIVATION(_result.s3, _result.s3); \
        ACTIVATION(_result.s4, _result.s4); \
        ACTIVATION(_result.s5, _result.s5); \
        ACTIVATION(_result.s6, _result.s6); \
        ACTIVATION(_result.s7, _result.s7); \
}
        RELU_8(_data0);
        RELU_8(_data1);

        intel_sub_group_block_write8((__global uint*)pDst + out_id, as_uint8(_data0));
        intel_sub_group_block_write8((__global uint*)pDst + out_id + 8 * batch_num, as_uint8(_data1));
    )__CC";




    const char convolution_code_yxfb_yxoi_B8_F8_memory[] = R"__CC(
        const __global float* input = (const __global float*)get_data(input_mem);
        __global float* pDst = (__global float*)get_data(dst_mem);

        const __global float* filter = (const __global float*)get_data(filter_mem);
        const __global float* bias = (const __global float*)get_data(bias_mem);

        const int batch_num = INPUT_BATCH_NUM;

        const uint linear_id_xy = get_global_id(1) + get_global_size(1) * get_global_id(2);
        const uint global_id = get_global_id(0) + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * FILTER_OUTPUT_FEATURE_NUM * batch_num;

        const uint out_batch_id = get_local_id(0);
        const uint out_fm = get_global_id(0) / INPUT_BATCH_NUM;
        const uint out_x = get_global_id(1);
        const uint out_y = get_global_id(2);

        const int ofm_offset = out_fm % FILTER_OUTPUT_FEATURE_NUM;

        float result = bias[ofm_offset];

        bool finish = false;

        finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
        finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

        float8 _data = 0.f;

        const uint sub_group_id = get_local_id(0);

        if(!finish)
        {
            const int x = out_x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
            const int y = out_y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

            for (uint i = 0; i < FILTER_SIZE_Y; i++)
            {
                const int input_offset_y = y + i;
                const bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;

                if(!zero_y)
                {
                    for (uint j = 0; j < FILTER_SIZE_X; j++)
                    {
                        const int input_offset_x = x + j;
                    
                        const bool zero = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;
    
                        if(!zero)
                        {
                            int input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * INPUT_FEATURE_NUM * batch_num;
                            input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * batch_num;
                            input_idx += out_batch_id;
                    
                            const int filter_idx = sub_group_id + FILTER_INPUT_FEATURE_NUM * ( ofm_offset +  FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));

                            for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h+=8)
                            {
                                float f_val = filter[filter_idx + h];
                                float8 _input = as_float8(intel_sub_group_block_read8((const __global uint*)input + input_idx + h * batch_num));
                                _data.s0 = fma(_input.s0, intel_sub_group_shuffle(f_val, 0), _data.s0);
                                _data.s1 = fma(_input.s1, intel_sub_group_shuffle(f_val, 1), _data.s1);                                
                                _data.s2 = fma(_input.s2, intel_sub_group_shuffle(f_val, 2), _data.s2);
                                _data.s3 = fma(_input.s3, intel_sub_group_shuffle(f_val, 3), _data.s3);
                                _data.s4 = fma(_input.s4, intel_sub_group_shuffle(f_val, 4), _data.s4);
                                _data.s5 = fma(_input.s5, intel_sub_group_shuffle(f_val, 5), _data.s5);
                                _data.s6 = fma(_input.s6, intel_sub_group_shuffle(f_val, 6), _data.s6);
                                _data.s7 = fma(_input.s7, intel_sub_group_shuffle(f_val, 7), _data.s7);
                            }
                        }
                    } 
                }
            }
        }
        result += _data.s0 + _data.s1 + _data.s2 + _data.s3 +
                  _data.s4 + _data.s5 + _data.s6 + _data.s7;

	ACTIVATION(pDst[global_id], result);
    )__CC";
}