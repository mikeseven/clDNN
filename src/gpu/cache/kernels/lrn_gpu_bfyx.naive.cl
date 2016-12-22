#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if FP16_UNIT_USED
    #define UNIT_CVT_FUNC(val) convert_half(val)
#else
    #define UNIT_CVT_FUNC(val) (val)
#endif


KERNEL (lrn_gpu_bfyx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint b_f = get_global_id(2);
    const uint b = b_f / INPUT_FEATURE_NUM;
    const uint f = b_f % INPUT_FEATURE_NUM;

    const uint linear_id = x + INPUT_SIZE_X * (y + INPUT_SIZE_Y * b_f);
    UNIT_TYPE acc = UNIT_VAL_ZERO;

    int input_offset_f = f + HELP_INPUT_OFFSET;
    int input_idx = x + INPUT_SIZE_X * (y + INPUT_SIZE_Y * (input_offset_f + INPUT_FEATURE_NUM * b));
    for (int i = 0; i < P_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT_FEATURE_NUM;

        UNIT_TYPE value = zero ? UNIT_VAL_ZERO : UNIT_CVT_FUNC(ALPHA_VAL_FACTOR) * input[input_idx];
        acc = mad(value, value, acc);

        input_offset_f++;
        input_idx += INPUT_SIZE_X * INPUT_SIZE_Y;
    }
    acc = mad(acc, UNIT_CVT_FUNC(ALPHA), UNIT_CVT_FUNC(K));
    acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));

    output[linear_id] = acc * input[linear_id];
}


#undef UNIT_CVT_FUNC