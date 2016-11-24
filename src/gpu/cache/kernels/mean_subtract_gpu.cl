#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL (mean_subtract_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, const __global UNIT_TYPE* mean)
{
    const uint batch_num = INPUT_BATCH_NUM;

    const uint global_id = get_global_id(0);
    const uint batch_id = global_id % batch_num;
    const uint feature_id = (global_id / batch_num) % INPUT_FEATURE_NUM;
    const uint x = ((global_id / batch_num) / INPUT_FEATURE_NUM) % INPUT_SIZE_X;
    const uint y = ((global_id / batch_num) / INPUT_FEATURE_NUM) / INPUT_SIZE_X;

#if BFYX_MEAN_FORMAT_USED
    // BFYX format of mean
    output[global_id] = input[global_id] - mean[x + MEAN_SIZE_X * (y + MEAN_SIZE_Y * (feature_id + MEAN_FEATURE_NUM * (batch_id % MEAN_BATCH_NUM)))];
#else
    // YXFB format of mean
    output[global_id] = input[global_id] - mean[(batch_id % MEAN_BATCH_NUM) + MEAN_BATCH_NUM * (feature_id + MEAN_FEATURE_NUM * (x + MEAN_SIZE_X * y))];
#endif
}