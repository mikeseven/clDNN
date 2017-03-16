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

    const uint element_group_offset = element_group_id * WORK_GROUP_SIZE * ELEMENTS_PER_WORK_ITEM;

    uint input_offset = element_group_offset + batch_id * INPUT_FEATURE_NUM * INPUT_SIZE_X * INPUT_SIZE_Y;
    uint output_offset = element_group_offset + batch_id * OUTPUT_FEATURE_NUM * OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
    output_offset += OUTPUT_SIZE_X * OUTPUT_SIZE_Y * depth_offset;

    if(element_group_offset < INPUT_ELEMENTS_COUNT - WORK_GROUP_SIZE * ELEMENTS_PER_WORK_ITEM )
    {
        float8 in = as_float8(intel_sub_group_block_read8((const __global uint*)input + input_offset));
        intel_sub_group_block_write8((__global uint*)output + output_offset, as_uint8(in));
    }
    else
    {
        // This is the last SIMD that needs to write only partial data.
        uint element_offset = get_global_id(1) * ELEMENTS_PER_WORK_ITEM;
        uint element_offset_in_workitem = element_offset - element_group_offset;
        for(uint i = 0; i < ELEMENTS_PER_WORK_ITEM; i++)
        {
            if(element_offset >= INPUT_ELEMENTS_COUNT)
                return;

            output[output_offset + element_offset_in_workitem] = input[input_offset + element_offset_in_workitem];
            element_offset_in_workitem++;
        }
    }
}

#undef WORK_GROUP_SIZE
#undef ELEMENTS_PER_WORK_ITEM