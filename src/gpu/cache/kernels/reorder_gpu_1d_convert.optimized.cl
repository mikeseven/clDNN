#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define TYPE_CVT_FUNC3(val, type) convert_##type(val)
#define TYPE_CVT_FUNC2(val, type) TYPE_CVT_FUNC3(val, type)
#if SRC_DEST_TYPE_CVT
    #define SRC_DEST_TYPE_CVT_FUNC(val) TYPE_CVT_FUNC2(val, DEST_TYPE)
#else
    #define SRC_DEST_TYPE_CVT_FUNC(val) val
#endif

KERNEL (reorder_gpu_1d_convert)(const __global SRC_TYPE* input, __global DEST_TYPE* output)
{
    const uint pos = get_global_id(2);

    output[pos] = SRC_DEST_TYPE_CVT_FUNC(input[pos]);
}

#undef SRC_DEST_TYPE_CVT_FUNC
#undef TYPE_CVT_FUNC2
#undef TYPE_CVT_FUNC3