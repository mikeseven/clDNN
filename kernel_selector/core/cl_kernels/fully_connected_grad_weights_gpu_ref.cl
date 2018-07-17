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

KERNEL(fully_connected_grad_weights_gpu_ref)(
    const __global INPUT0_TYPE* input_grad,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global UNIT_TYPE* bias,
#endif
#if MOMENTUM
    __global UNIT_TYPE* prev_grad_w,
#if BIAS_TERM
    __global UNIT_TYPE* prev_grad_b,
#endif
#endif
    const __global INPUT1_TYPE* input,
    const float lr
    )
{
    const uint ofm_ifm       = get_global_id(0);
    const uint id_x          = (uint)get_global_id(1);
    const uint id_y          = (uint)get_global_id(2);
    const uint ifm           = ofm_ifm % FILTER_IFM_NUM;
    const uint ofm           = ofm_ifm / FILTER_IFM_NUM;

    ACCUMULATOR_TYPE grad_w = 0;
#if BIAS_TERM
    ACCUMULATOR_TYPE grad_b = 0;
#endif

    const uint filter_idx = GET_FILTER_INDEX(FILTER, ofm, ifm, id_y, id_x);
    for (uint b = 0; b < INPUT0_BATCH_NUM; b++)
    {
        const uint input_grad_idx = GET_DATA_INDEX(INPUT0, b, 0, 0, ofm);
        const uint input_idx = GET_DATA_INDEX(INPUT1, b, ifm, id_y, id_x);
        UNIT_TYPE grad = input_grad[input_grad_idx];
        grad_w += input[input_idx] * grad;
#if BIAS_TERM
        grad_b += grad;
#endif
    }

#if MOMENTUM
    UNIT_TYPE update_gradient_w = lr * (grad_w + prev_grad_w[filter_idx] * ALPHA) + DECAY_RATE * lr * weights[filter_idx];
    weights[filter_idx] -= update_gradient_w;
    prev_grad_w[filter_idx] = update_gradient_w;
#else
    weights[filter_idx] -= lr * grad_w + DECAY_RATE * lr * weights[filter_idx];
#endif

#if BIAS_TERM
    if(ifm == 0 && id_x == 0 && id_y == 0)
    {
#if MOMENTUM
        UNIT_TYPE update_gradient_b = lr * (prev_grad_b[ofm] * ALPHA + grad_b);
        bias[ofm] -= update_gradient_b;
        prev_grad_b[ofm] = update_gradient_b;
#else
        bias[ofm] -= lr * grad_b;
#endif
    }
#endif
    

}