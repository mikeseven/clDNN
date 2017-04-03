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

//
// In this kernel we are processing "fyx" as flatten 1D "elements".
// As long as we can we use block read/write.
// For last SIMD in which we have to write only partial data we use normal read/write to buffer.
//

// must be 8 as long as we use block_read8/write8
#define ELEMENTS_PER_WORK_ITEM 8

#define WORK_GROUP_SIZE 16

__attribute__((reqd_work_group_size(1, WORK_GROUP_SIZE, 1)))
__attribute__((intel_reqd_sub_group_size(WORK_GROUP_SIZE)))
KERNEL (depth_concatenate_gpu_bfyx_no_padding)(__global float* input, __global float* output, uint depth_offset)
{
    const uint batch_id = get_group_id(0);

    // Which pack of 16*8 elements we are processing.
    uint element_group_id = get_group_id(1);
    uint element_offset = (uint)get_global_id(1) * ELEMENTS_PER_WORK_ITEM;

    const uint element_group_offset = element_group_id * WORK_GROUP_SIZE * ELEMENTS_PER_WORK_ITEM;

    uint input_offset = element_group_offset + batch_id * INPUT_FEATURE_NUM * INPUT_SIZE_X * INPUT_SIZE_Y;
    uint output_batch_offset = batch_id * OUTPUT_FEATURE_NUM * OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
    uint output_offset = element_group_offset + output_batch_offset;
    uint depth_offset_size = OUTPUT_SIZE_X * OUTPUT_SIZE_Y * depth_offset;
    output_offset += depth_offset_size;

    //Check if current group in batch starts from 16-byte aligned pos. If not then move block read to 16-byte aligned position.
    //Requirement for intel_sub_group_block_write8.
    uint align_offset = 0;
    uint group_start_pos = element_group_offset + output_batch_offset + depth_offset_size;
    if(group_start_pos % WORK_GROUP_SIZE != 0)
    {
        uint next_aligned_pos = group_start_pos / WORK_GROUP_SIZE * WORK_GROUP_SIZE + WORK_GROUP_SIZE;
        align_offset = next_aligned_pos - group_start_pos;
    }

    if(element_group_offset + align_offset + WORK_GROUP_SIZE * ELEMENTS_PER_WORK_ITEM < INPUT_ELEMENTS_COUNT)
    {
        float8 in = as_float8(intel_sub_group_block_read8((const __global uint*)input + input_offset + align_offset));
        intel_sub_group_block_write8((__global uint*)output + output_offset + align_offset, as_uint8(in));
        
        //Fill the values that were missed upon adding align_offset
        if((align_offset != 0) && (element_offset + output_batch_offset < group_start_pos + align_offset))
        {
            for(uint i = 0; i < align_offset; i++)
                output[output_offset + i] = input[input_offset + i];
        }
    }
    else
    {
        // This is the last SIMD that needs to write only partial data.
        uint element_offset_in_workitem = element_offset - element_group_offset;
        for(uint i = 0; i < ELEMENTS_PER_WORK_ITEM; i++)
        {
            if(element_offset + i >= INPUT_ELEMENTS_COUNT)
                return;

            output[output_offset + element_offset_in_workitem] = input[input_offset + element_offset_in_workitem];
            element_offset_in_workitem++;
        }
    }
}

#undef WORK_GROUP_SIZE
#undef ELEMENTS_PER_WORK_ITEM