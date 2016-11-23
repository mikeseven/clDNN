// Move this to common program header once we unify ACTIVATION among all kernels.
// ---- START ----
#define TYPE_CVT_FUNC3(val, type) convert_##type(val)
#define TYPE_CVT_FUNC2(val, type) TYPE_CVT_FUNC3(val, type)
#define MAKE_FP_LITERAL3(val, suffix) val##suffix
#define MAKE_FP_LITERAL2(val, suffix) MAKE_FP_LITERAL3(val, suffix)
// ---- END ----

#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if FP16_UNIT_USED
    #define MAKE_FP_LITERAL(val) MAKE_FP_LITERAL2(val, UNIT_SUFFIX)
    #define UNIT_CVT_FUNC(val) TYPE_CVT_FUNC2(val, UNIT_TYPE)
#else
    #define MAKE_FP_LITERAL(val) MAKE_FP_LITERAL2(val, f)
    #define UNIT_CVT_FUNC(val) val
#endif

#ifdef RELU
    #define ACTIVATION(output, input) output = max(input, MAKE_FP_LITERAL(0.0)) + UNIT_CVT_FUNC(NEGATIVE_SLOPE) * min(input, MAKE_FP_LITERAL(0.0));
#else
    #define ACTIVATION(output, input) output = input;
#endif

// Required JIT constants:
//  - FP16_SUPPORTED       - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED       - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE            - Type of unit of input/output/weight/bias.
//  - UNIT_SUFFIX          - Suffix for floating-point literals used by current UNIT_TYPE.
//  - INPUT_BATCH_NUM      - [int] Number of elements from single spatial and single feature that are grouped in single batch in input.
//  - INPUT_ELEMENTS_COUNT - [int] Cumulative number of elements from input that are processed in single batch.
//  - WEIGHTS_BATCH_NUM    - [int] Cumulative number of elements that are outputted in single batch.
// Optional JIT constants:
//  - RELU           - Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE - [float] Factor for negative output values (required when RELU is specified).

KERNEL (fully_connected_gpu_xb_xb)(
    const __global UNIT_TYPE* input, 
    __global UNIT_TYPE* output, 
    const __global UNIT_TYPE* weight,
    const __global UNIT_TYPE* bias)
{
    const uint x = get_global_id(0);
    const uint batch_id = x % INPUT_BATCH_NUM;

    const uint outXIdx = x / INPUT_BATCH_NUM;
    UNIT_TYPE result = MAKE_FP_LITERAL(0.0);

    uint input_idx = batch_id;
    uint weight_idx = outXIdx;
    for (uint i = 0; i < INPUT_ELEMENTS_COUNT; i++)
    {
        result += input[input_idx] * weight[weight_idx];
        input_idx += INPUT_BATCH_NUM;
        weight_idx += WEIGHTS_BATCH_NUM;
    }

    result += bias[outXIdx];
    ACTIVATION(output[x], result);
}

#undef ACTIVATION

#undef UNIT_CVT_FUNC
#undef MAKE_FP_LITERAL

// Move this to common program header once we unify ACTIVATION among all kernels.
// ---- START ----
#undef MAKE_FP_LITERAL2
#undef MAKE_FP_LITERAL3
#undef TYPE_CVT_FUNC2
#undef TYPE_CVT_FUNC3
// ---- END ----