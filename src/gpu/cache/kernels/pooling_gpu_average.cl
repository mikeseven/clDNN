#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL(pooling_gpu_average)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint linear_id_xyz = get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2));

    const int offset_x = get_global_id(1) * STRIDE_SIZE_X;
    const int offset_y = get_global_id(2) * STRIDE_SIZE_Y;

    UNIT_TYPE result = UNIT_INIT_VAL_AVG;

    const int batch_and_feature_offset = get_global_id(0);
    int input_idx = batch_and_feature_offset + OUTPUT_BATCH_NUM * INPUT_FEATURE_NUM * (offset_x + offset_y * INPUT_SIZE_X);
    for(uint j = 0; j < WINDOW_SIZE_Y; j++)
    {
        for(uint i = 0; i < WINDOW_SIZE_X; i++)
        {
            result += input[input_idx];
            input_idx += OUTPUT_BATCH_NUM * INPUT_FEATURE_NUM;
        }
        input_idx += OUTPUT_BATCH_NUM * INPUT_FEATURE_NUM * (INPUT_SIZE_X - WINDOW_SIZE_X);
    }
    output[linear_id_xyz] = result / (UNIT_TYPE)(WINDOW_SIZE_Y * WINDOW_SIZE_X);
}