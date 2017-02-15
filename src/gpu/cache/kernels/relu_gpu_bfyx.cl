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


KERNEL (relu_gpu_bfyx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{    
    const uint global_id = get_global_id(0);
    const uint batch_id = global_id % batch_num;
    const uint feature_id = (global_id / batch_num) % INPUT_FEATURE_NUM;
    const uint x = ((global_id / batch_num) / INPUT_FEATURE_NUM) % INPUT_SIZE_X;
    const uint y = ((global_id / batch_num) / INPUT_FEATURE_NUM) / INPUT_SIZE_X;

    uint input_id = batch_id * INPUT_FEATURE_NUM * (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X);
    input_id += feature_id * (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X);
    input_id += (INPUT_PADDING_SIZE_Y + y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X);
    input_id += INPUT_PADDING_SIZE_X + x;

    uint output_id = batch_id * OUTPUT_FEATURE_NUM * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X);
    output_id += feature_id * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X);
    output_id += (y + OUTPUT_PADDING_SIZE_Y) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X);
    output_id += x + OUTPUT_PADDING_SIZE_X;

    ACTIVATION(output[output_id], input[input_id]);
}

#undef ACTIVATION
