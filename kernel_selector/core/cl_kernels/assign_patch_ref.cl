/*
// Copyright (c) 2017 Intel Corporation
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

#include "include/include_all.cl"

inline void atomicAdd(volatile __global float *addr, float val)
{
    union{
        unsigned int uintVal;
        float        floatVal;
    } next, current, tmp;

    current.floatVal    = *addr;
    do
    {
        tmp.floatVal = current.floatVal;
        next.floatVal = tmp.floatVal + val;
        current.uintVal  = atomic_cmpxchg( (volatile __global unsigned int *)addr, tmp.uintVal, next.uintVal);
    } while( current.uintVal != tmp.uintVal );
}

KERNEL (assign_patch_gpu_ref)(__global UNIT_TYPE* input, __global UNIT_TYPE* nn, __global UNIT_TYPE* output)
{
    const uint f = get_global_id(0);
    const uint x = get_global_id(1);
    const uint y = get_global_id(2);
    
    uint output_idx = f*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
    uint input_idx = nn[y*INPUT1_Y_PITCH + x*INPUT1_X_PITCH] * INPUT0_BATCH_PITCH + f*INPUT0_FEATURE_PITCH;

    for (int i = 0; i < INPUT0_SIZE_X; i++){
        for (int j = 0; j < INPUT0_SIZE_Y; j++){
            atomicAdd(output + output_idx, input[input_idx]);
            output_idx++;
            input_idx++;
        }
        output_idx += OUTPUT_SIZE_Y - INPUT0_SIZE_Y;
    }
}
