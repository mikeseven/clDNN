#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL(pooling_gpu_bfyx_average_offset)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    // constexpr:
    const int input_buffer_size_x = INPUT_PADDING_LOWER_SIZE_X + INPUT_SIZE_X + INPUT_PADDING_UPPER_SIZE_X;
    const int input_buffer_size_y = INPUT_PADDING_LOWER_SIZE_Y + INPUT_SIZE_Y + INPUT_PADDING_UPPER_SIZE_Y;
    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;


    const uint linear_id_xyz = get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2));

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const int offset_x = INPUT_PADDING_LOWER_SIZE_X + x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
    const int offset_y = INPUT_PADDING_LOWER_SIZE_Y + y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

    UNIT_TYPE result = UNIT_INIT_VAL_AVG;

    const int batch_and_feature_offset = get_global_id(2);
    for(uint j = 0; j < WINDOW_SIZE_Y; j++)
    {
        int input_offset_y = offset_y + j;
        bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;
        if(!zero_y)
        {
            for(uint i = 0; i < WINDOW_SIZE_X; i++)
            {
                int input_offset_x = offset_x + i;
                bool zero = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;
                if(!zero)
                {
                    int input_idx = batch_and_feature_offset * input_buffer_size_x * input_buffer_size_y;
                    input_idx += input_offset_y * input_buffer_size_x + input_offset_x;
                    result += input[input_idx];
                }
            }
        }
    }

    const uint b = batch_and_feature_offset / INPUT_FEATURE_NUM;
    const uint f = batch_and_feature_offset % INPUT_FEATURE_NUM;
    uint output_pos = (b * OUTPUT_FEATURE_NUM + f) * output_buffer_size_x * output_buffer_size_y;
    output_pos += (OUTPUT_PADDING_LOWER_SIZE_Y + y) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X + x;

    output[output_pos] = result / (UNIT_TYPE)(WINDOW_SIZE_Y * WINDOW_SIZE_X);
}