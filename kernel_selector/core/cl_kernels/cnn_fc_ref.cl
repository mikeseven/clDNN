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

KERNEL(fc)(
    __global DATA_TYPE* input, 
    __global DATA_TYPE* output, 
    __global DATA_TYPE* weights, 
    __global DATA_TYPE* biases)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
#if OUT_BATCH == 1
    const unsigned z = get_global_id(2);
    const unsigned w = 0;
#else
    const unsigned z = get_global_id(2) % OUT_DEPTH;
    const unsigned w = get_global_id(2) / OUT_DEPTH;
#endif
    
    const unsigned int input_size = INPUT_WIDTH * INPUT_HEIGHT * INPUT_DEPTH;

    unsigned int output_idx = w*OUT_BATCH_PITCH + z*OUT_FEATURE_PITCH + y * OUT_Y_PITCH + x + OUT_OFFSET;
    unsigned offset = z*OUT_WIDTH * OUT_HEIGHT + y*OUT_WIDTH + x;
    COUNTER_TYPE dotProd = (COUNTER_TYPE)(biases[offset]);

    __global DATA_TYPE* processed_neuron_weights = weights + offset * input_size;
    __global DATA_TYPE* processed_input_batch =  input;
    unsigned int weight_idx =0;

    for (unsigned int plane = 0; plane < INPUT_DEPTH; ++plane)
    {
       for (unsigned int height = 0; height < INPUT_HEIGHT; ++height)
       {
           for(unsigned int width = 0; width < INPUT_WIDTH; ++width )
           {
               unsigned int input_idx = w*INPUT_BATCH_PITCH + plane*INPUT_FEATURE_PITCH + height*INPUT_Y_PITCH + width + INPUT_OFFSET;

               dotProd += (COUNTER_TYPE)(processed_input_batch[input_idx] * processed_neuron_weights[weight_idx]);

               weight_idx++;
          }
       }
    }
    output[output_idx] = FUNC_CALL(activation_function)((DATA_TYPE)dotProd, NL_M, NL_N);
}