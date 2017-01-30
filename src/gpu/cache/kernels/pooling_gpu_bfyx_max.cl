#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL(pooling_gpu_bfyx_max)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint linear_id_xyz = get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2));

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const int offset_x = x * STRIDE_SIZE_X;
    const int offset_y = y * STRIDE_SIZE_Y;

    UNIT_TYPE result = UNIT_INIT_VAL_MAX;

    const uint batch_and_feature_offset = get_global_id(2);
    int input_idx = batch_and_feature_offset * INPUT_SIZE_X * INPUT_SIZE_Y + offset_y * INPUT_SIZE_X + offset_x;
    for(uint j = 0; j < WINDOW_SIZE_Y; j++)
    {
        for(uint i = 0; i < WINDOW_SIZE_X; i++)
        {
            result = max(result, input[input_idx]);
            input_idx++;
        }
        input_idx += (INPUT_SIZE_X - WINDOW_SIZE_X);
    }

    const uint b = batch_and_feature_offset / INPUT_FEATURE_NUM;
    const uint f = batch_and_feature_offset % INPUT_FEATURE_NUM;
    uint output_pos = b * OUTPUT_FEATURE_NUM * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X);
    output_pos += f * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X);
    output_pos += (y + OUTPUT_PADDING_SIZE_Y) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X);
    output_pos += x + OUTPUT_PADDING_SIZE_X;
    output[output_pos] = result;
}