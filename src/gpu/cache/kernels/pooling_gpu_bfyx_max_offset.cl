#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL(pooling_gpu_bfyx_max_offset)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint linear_id_xyz = get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2));

    const int offset_x = get_global_id(0) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
    const int offset_y = get_global_id(1) * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

    UNIT_TYPE result = UNIT_INIT_VAL_MAX;

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
                    int input_idx = batch_and_feature_offset * INPUT_SIZE_X * INPUT_SIZE_Y + input_offset_y * INPUT_SIZE_X + input_offset_x;
                    result = max(result, input[input_idx]);
                }
            }
        }

    }

    if (offset_y < 0 || offset_y + WINDOW_SIZE_Y - 1 >= INPUT_SIZE_Y || offset_x < 0 || offset_x + WINDOW_SIZE_X - 1 >= INPUT_SIZE_X)
        output[linear_id_xyz] = max(result, (UNIT_TYPE)0);
    else
        output[linear_id_xyz] = result;
}