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


#include "include/include_all.cl"

KERNEL (concatenation_gpu_ref)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint output_offset_in_concat_axis)
{
    const uint d1 = get_global_id(0);
    const uint d2 = get_global_id(1);
#ifdef CHECK_FEATURES
    if(d2 >= INPUT0_FEATURES)
        return;
#endif
    const uint d3 = get_global_id(2);
    
    uint input_offset  = INPUT0_OFFSET + d1*INPUT0_PITCHES[1] + d2*INPUT0_PITCHES[2] + d3*INPUT0_PITCHES[3];
    uint output_offset = OUTPUT_OFFSET + d1*OUTPUT_PITCHES[1] + d2*OUTPUT_PITCHES[2] + d3*OUTPUT_PITCHES[3] + output_offset_in_concat_axis*OUTPUT_PITCHES[CONCAT_AXIS_INDEX];

    for (size_t idx = 0; idx < INPUT0_SIZES[0]; ++idx)
    {
        output[output_offset] = input[input_offset];
        input_offset  += INPUT0_PITCHES[0];
        output_offset += OUTPUT_PITCHES[0];
    }
}
