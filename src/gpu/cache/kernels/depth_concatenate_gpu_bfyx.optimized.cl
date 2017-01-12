#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL (depth_concatenate_gpu_bfyx)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint depth_offset)
{
    uint global_id = get_global_id(0);

    uint input_offset = global_id;
    uint output_offset = global_id + OUTPUT_SIZE_X * OUTPUT_SIZE_Y * depth_offset;
    for(uint f = 0; f < INPUT_FEATURE_NUM * OUTPUT_BATCH_NUM; f++)
    {
        output[output_offset] = input[input_offset];
        output_offset += OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
        input_offset += INPUT_SIZE_X * INPUT_SIZE_Y;
    }
}