#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if RELU && FP16_UNIT_USED
    #define ACTIVATION(output, input) output = max(input, 0.0h) + convert_half(NEGATIVE_SLOPE) * min(input, 0.0h);
#elif RELU
    #define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
    #define ACTIVATION(output, input) output = input;
#endif

KERNEL (eltwise_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, const __global UNIT_TYPE* input2)
{
    const uint batch_num = INPUT_BATCH_NUM;

    const uint global_id = get_global_id(0);
    const uint batch_id = global_id % batch_num;
    const uint feature_id = (global_id / batch_num) % INPUT_FEATURE_NUM;
    const uint x = ((global_id / batch_num) / INPUT_FEATURE_NUM) % INPUT_SIZE_X;
    const uint y = ((global_id / batch_num) / INPUT_FEATURE_NUM) / INPUT_SIZE_X;

    const UNIT_TYPE in1 = input[global_id]; 
    const UNIT_TYPE in2 = input2[(batch_id % INPUT2_BATCH_NUM) + INPUT2_BATCH_NUM * (feature_id + INPUT2_FEATURE_NUM * (x + INPUT2_SIZE_X * y))];
    UNIT_TYPE result;
#if MAX_MODE_USED
    result = (in1 > in2 ? in1 : in2);
#elif PROD_MODE_USED
    result = in1 * in2;
#elif SUB_MODE_USED
    result = in1 - in2;
#else 
    result = in1 + in2;
#endif

    ACTIVATION(output[global_id], result);
}