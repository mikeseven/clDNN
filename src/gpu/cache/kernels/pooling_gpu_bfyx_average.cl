#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL(pooling_gpu_bfyx_average)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint linear_id_xyz = get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2));

    const int offset_x = get_global_id(0) * STRIDE_SIZE_X;
    const int offset_y = get_global_id(1) * STRIDE_SIZE_Y;

    UNIT_TYPE result = UNIT_INIT_VAL_AVG;

    const int batch_and_feature_offset = get_global_id(2);
    int input_idx = batch_and_feature_offset * INPUT_SIZE_X * INPUT_SIZE_Y + offset_y * INPUT_SIZE_X + offset_x;
    for(uint j = 0; j < WINDOW_SIZE_Y; j++)
    {
        for(uint i = 0; i < WINDOW_SIZE_X; i++)
        {
            result += input[input_idx];
            input_idx++;
        }
        input_idx += (INPUT_SIZE_X - WINDOW_SIZE_X);
    }
    output[linear_id_xyz] = result / (UNIT_TYPE)(WINDOW_SIZE_Y * WINDOW_SIZE_X);
}