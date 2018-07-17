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

#include "include/include_all.cl"

#define DECAY_RATE 0.0005f
#define ALPHA 0.9f

KERNEL(convolution_grad_weights_gpu_ref)(
    const __global UNIT_TYPE* input_grad,
    __global UNIT_TYPE* output,
    __global UNIT_TYPE* filter,
#if BIAS_TERM
    __global UNIT_TYPE* bias,
#endif
#if MOMENTUM
    __global UNIT_TYPE* prev_grad_w,
#if BIAS_TERM
    __global UNIT_TYPE* prev_grad_b,
#endif
#endif
    const __global UNIT_TYPE* input,
    uint split_idx,
    float lr)
{
    const uint ofm_ifm       = get_global_id(0);
    const uint id_x          = (uint)get_global_id(1);
    const uint id_y          = (uint)get_global_id(2);
    const uint ifm           = ofm_ifm % INPUT1_FEATURE_NUM;
    const uint ofm           = ofm_ifm / INPUT1_FEATURE_NUM;

    const int in_x    = id_x - PADDING_SIZE_X;
    const int in_y    = id_y - PADDING_SIZE_Y;

    ACCUMULATOR_TYPE grad_w = 0;
    ACCUMULATOR_TYPE grad_b = 0;

    uint weights_idx = ofm * FILTER_OFM_PITCH + ifm * FILTER_IFM_PITCH + id_y * FILTER_Y_PITCH + id_x * FILTER_X_PITCH;

    for(int b = 0; b < INPUT0_BATCH_NUM; b++)
    {
        UNIT_TYPE result = UNIT_VAL_ZERO;

#if BIAS_TERM
        UNIT_TYPE result_bias = UNIT_VAL_ZERO;
#endif

        const uint grad_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_OFM_NUM;
        const uint in_split_offset = split_idx * INPUT1_FEATURE_PITCH * FILTER_IFM_NUM;

        for (uint i = 0; i < INPUT0_SIZE_Y; i++)
        {
            for (uint j = 0; j < INPUT0_SIZE_X; j++)
            {
                const int input_offset_y = in_y + i * STRIDE_SIZE_Y;
                const bool zero_y = input_offset_y >= INPUT1_SIZE_Y || input_offset_y < 0;
                const int input_offset_x = in_x + j * STRIDE_SIZE_X;
                const bool zero_x = input_offset_x >= INPUT1_SIZE_X || input_offset_x < 0;
#if BIAS_TERM
                uint input_grad_idx = grad_split_offset + b*INPUT0_BATCH_PITCH + ofm*INPUT0_FEATURE_PITCH + j*INPUT0_X_PITCH + i*INPUT0_Y_PITCH;
                UNIT_TYPE grad = input_grad[input_grad_idx];
#endif
                if(!zero_x && !zero_y)
                {
                    uint input_idx = in_split_offset + b*INPUT1_BATCH_PITCH + ifm*INPUT1_FEATURE_PITCH + (uint)input_offset_x*INPUT1_X_PITCH + (uint)input_offset_y*INPUT1_Y_PITCH;
#if BIAS_TERM
                    result = fma(input[input_idx], grad, result);
#else
                    uint input_grad_idx = grad_split_offset + b*INPUT1_BATCH_PITCH + ofm*INPUT0_FEATURE_PITCH + j*INPUT0_X_PITCH + i*INPUT0_Y_PITCH;
                    result = fma(input[input_idx], input_grad[input_grad_idx], result);
#endif
                }
#if BIAS_TERM
                result_bias += grad;
#endif
            }
        }

        grad_w += result;

#if BIAS_TERM
        if(ifm == 0 && id_x == 0 && id_y == 0)
        {
            grad_b += result_bias;
        }
#endif
    }

#if MOMENTUM
    filter[weights_idx] -= lr * (prev_grad_w[weights_idx] * ALPHA + grad_w) + DECAY_RATE * lr * filter[weights_idx];
    prev_grad_w[weights_idx] = grad_w;
#else
    filter[weights_idx] -= lr * grad_w + DECAY_RATE * lr * filter[weights_idx];
#endif

#if BIAS_TERM
        if(ifm == 0 && id_x == 0 && id_y == 0)
        {
#if MOMENTUM
        bias[ofm] -= lr * (prev_grad_b[ofm] * ALPHA + grad_b);
        prev_grad_b[ofm] = grad_b;
#else
        bias[ofm] -= lr * grad_b;
#endif
        }
#endif

}
