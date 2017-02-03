#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL (depth_concatenate_gpu_bfyx)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint depth_offset)
{
    uint batch_id = get_global_id(0);
    uint feature_and_y_id = get_global_id(1);
    uint feature_id = feature_and_y_id / INPUT_SIZE_Y;
    uint y = feature_and_y_id % INPUT_SIZE_Y;

    if(feature_id >= INPUT_FEATURE_NUM)
        return;

    uint input_offset = batch_id * INPUT_FEATURE_NUM * (INPUT_SIZE_X + INPUT_PADDING_SIZE_X * 2) * (INPUT_SIZE_Y + INPUT_PADDING_SIZE_Y * 2);
    input_offset += feature_id * (INPUT_SIZE_X + INPUT_PADDING_SIZE_X * 2) * (INPUT_SIZE_Y + INPUT_PADDING_SIZE_Y * 2);
    input_offset += (INPUT_PADDING_SIZE_Y + y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X);
    input_offset += INPUT_PADDING_SIZE_X;

    uint output_offset = batch_id * OUTPUT_FEATURE_NUM * (OUTPUT_SIZE_X + OUTPUT_PADDING_SIZE_X * 2) * (OUTPUT_SIZE_Y + OUTPUT_PADDING_SIZE_Y * 2);
    output_offset += feature_id * (OUTPUT_SIZE_X + OUTPUT_PADDING_SIZE_X * 2) * (OUTPUT_SIZE_Y + OUTPUT_PADDING_SIZE_Y * 2);
    output_offset += (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y) * depth_offset;
    output_offset += (OUTPUT_PADDING_SIZE_Y + y) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X);
    output_offset += OUTPUT_PADDING_SIZE_X;
    
    for(uint x = 0; x < INPUT_SIZE_X; x++)
    {
        output[output_offset++] = input[input_offset++];
    }
}