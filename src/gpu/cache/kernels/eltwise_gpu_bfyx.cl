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

KERNEL (eltwise_gpu_bfyx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, const __global UNIT_TYPE* input2)
{
    // constexpr:
    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;


    const uint batch_num = INPUT_BATCH_NUM;

    const uint global_id = get_global_id(0);

    uint input_id = global_id;

#if OUTPUT_PADDING_LOWER_SIZE_X > 0 || OUTPUT_PADDING_UPPER_SIZE_X > 0 || OUTPUT_PADDING_LOWER_SIZE_Y > 0 || OUTPUT_PADDING_UPPER_SIZE_Y > 0
    const uint x = global_id % INPUT_SIZE_X;
    const uint y = (global_id / (INPUT_SIZE_X)) % INPUT_SIZE_Y;
    const uint f = (global_id / (INPUT_SIZE_X * INPUT_SIZE_Y)) % INPUT_FEATURE_NUM;
    const uint b = (global_id / (INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_FEATURE_NUM));
    
    uint output_id = (b * OUTPUT_FEATURE_NUM + f) * output_buffer_size_x * output_buffer_size_y;
    output_id += (OUTPUT_PADDING_LOWER_SIZE_Y + y) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X + x;
#else
    uint output_id = input_id;
#endif

    const UNIT_TYPE in1 = input[input_id]; 
    const UNIT_TYPE in2 = input2[input_id];
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

    ACTIVATION(output[output_id], result);
}

#undef ACTIVATION
