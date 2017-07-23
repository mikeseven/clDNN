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


#include "include/common.cl"

KERNEL(convolution_gpu_yxfb_yxio_b16)(
    const __global float* input,
    __global UNIT_TYPE* output,
    const __global float* filter,
#if BIAS_TERM
    const __global float* bias,
#endif
    uint split_idx)
{
    // get_global_size(0) -> Number of work items needed to compute all features and all batches for single output spatial position
    //                       (single (x, y) point in output).
    // get_global_size(1) -> Output size in X-dimension.
    // get_global_size(2) -> Output size in Y-dimension.
    // get_global_id(0)   -> Id of work item computing single spatial point of output indicated by get_global_id(1), get_global_id(2).
    // get_global_id(1)   -> Current x-position in output.
    // get_global_id(2)   -> Current y-position in output.
    //
    // WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS -> Number of work items needed to compute entire one batch for at least one feature and one spatial point.
    //                                           (this number in current implementation computes also OFM_PER_WORK_ITEM output features at the same time).
    // FILTER_ARRAY_NUM                       -> Number of filters groups (split size).


    const uint linear_id_xy = get_global_id(1) + get_global_size(1) * get_global_id(2);
    uint global_id = (((uint)get_global_id(0) / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS) + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * (FILTER_OUTPUT_FEATURE_NUM / OFM_PER_WORK_ITEM)) * WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS;

    const uint sub_group_id = get_local_id(0);

#if defined(USE_BLOCK_READ_2) || defined(USE_BLOCK_READ_1)
    const uint chunk_size = sizeof(uint)/sizeof(UNIT_TYPE);
#else
    const uint chunk_size = 1;
#endif

    const uint out_batch_id = chunk_size * sub_group_id + LOCAL_WORK_GROUP_SIZE * BATCHES_PER_WORK_ITEM * ((uint)get_group_id(0) % LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS);
    const uint out_x = get_global_id(1);
    const uint out_y = get_global_id(2);

    const uint out_id = (global_id / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS) * OFM_PER_WORK_ITEM * INPUT0_BATCH_NUM + out_batch_id;

    const uint ofm_offset = ((global_id * OFM_PER_WORK_ITEM) / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS) % FILTER_OUTPUT_FEATURE_NUM;

    // Each component of vector element contains computation for separate output feature.
    float8 _data[BATCHES_PER_WORK_ITEM];
    for(uint i = 0; i < BATCHES_PER_WORK_ITEM; i++)
    {
        _data[i] = UNIT_VAL_ZERO;
    }

    const int x = (int)out_x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int y = (int)out_y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    for (uint i = 0; i < FILTER_SIZE_Y; i++)
    {
        const int input_offset_y = y + i * DILATION_SIZE_Y;
        const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

        if(!zero_y)
        {
            for (uint j = 0; j < FILTER_SIZE_X; j++)
            {
                const int input_offset_x = x + j * DILATION_SIZE_X;
                const bool zero = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                if(!zero)
                {
                    uint input_idx = input_offset_x*INPUT0_X_PITCH + input_offset_y*INPUT0_Y_PITCH;
                    input_idx += INPUT0_OFFSET + split_idx * FILTER_INPUT_FEATURE_NUM * INPUT0_FEATURE_PITCH;
                    input_idx += out_batch_id;

                    //sub_group_id used as offset to make each workitem load different filter, and then shuffle it
                    uint filter_idx = ofm_offset + sub_group_id + i*FILTER_Y_PITCH + j*FILTER_X_PITCH;

                    for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
                    {
#ifdef USE_BLOCK_READ_2
                        float2 _input = as_float2(intel_sub_group_block_read2((const __global uint*)input + input_idx));
                        float8 filter_transp = TRANSPOSE_BLOCK_8(filter[filter_idx]);
                        _data[0] = fma(_input.s0, filter_transp, _data[0]);
                        _data[1] = fma(_input.s1, filter_transp, _data[1]);
                        input_idx += INPUT0_FEATURE_PITCH;
#else
                        float8 filter_transp = TRANSPOSE_BLOCK_8(filter[filter_idx]);
                        for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
                        {
                            _data[s] = fma(input[input_idx], filter_transp, _data[s]);
                            input_idx += LOCAL_WORK_GROUP_SIZE;
                        }
                        input_idx += INPUT0_FEATURE_PITCH - BATCHES_PER_WORK_ITEM * LOCAL_WORK_GROUP_SIZE;
#endif
                        filter_idx += FILTER_IFM_PITCH;
                    }
                }
            }
        }
    }

#if BIAS_TERM
    float bias_val = bias[ofm_offset + sub_group_id];
    for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
    {
        ADD_BIAS_8(_data[s], bias_val);
    }
#endif
    for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
    {
        _data[s] = ACTIVATION(_data[s], NL_M, NL_N);
    }

    for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
    {
        uint _out_id = OUTPUT_OFFSET + out_id + s * LOCAL_WORK_GROUP_SIZE;
        output[_out_id] = _data[s].s0; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s1; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s2; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s3; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s4; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s5; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s6; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s7; _out_id += OUTPUT_FEATURE_PITCH;
    }
}
