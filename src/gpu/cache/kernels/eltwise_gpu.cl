#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if RELU && FP16_UNIT_USED
    #define ACTIVATION(output, input) output = isinf(convert_half(NEGATIVE_SLOPE)) ? ((input >= 0.0h) ? \
    input : -convert_half(NEGATIVE_SLOPE)) : (max(input, 0.0h) + convert_half(NEGATIVE_SLOPE) * min(input, 0.0h));
#elif RELU
    #define ACTIVATION(output, input) output = isinf(NEGATIVE_SLOPE) ? ((input >= 0.0f) ? \
    input : -NEGATIVE_SLOPE) : (max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f));
#else
    #define ACTIVATION(output, input) output = input;
#endif

KERNEL (eltwise_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, const __global UNIT_TYPE* input2)
{
    const uint global_id = get_global_id(0);

    const UNIT_TYPE in1 = input[global_id]; 
    const UNIT_TYPE in2 = input2[global_id];
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

#undef ACTIVATION
