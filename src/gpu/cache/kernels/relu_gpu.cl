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


KERNEL (relu_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{    
    const uint global_id = get_global_id(0);
    ACTIVATION(output[global_id], input[global_id]);
}

#undef ACTIVATION