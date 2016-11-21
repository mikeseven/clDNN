#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


UNIT_TYPE FUNC(find_max_value)(__local UNIT_TYPE* partial_max, const int idx, const __global UNIT_TYPE* input)
{
    UNIT_TYPE value = -UNIT_VAL_MAX;
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        value = max(value, input[LWS * i + idx]);
    }
    value = max(value, idx < LEFTOVERS? input[LWS * ITEMS_NUM + idx] : -UNIT_VAL_MAX);
    partial_max[idx] = value;

    barrier(CLK_LOCAL_MEM_FENCE);
    if(idx == 0)
    {
        for(int i = 1; i < LWS; i++)
        {
            partial_max[0] = max(partial_max[0], partial_max[i]);
        };
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return partial_max[0];
}

KERNEL (softmax_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const int idx = get_local_id(0);

    __local UNIT_TYPE partial_max[LWS];
    const UNIT_TYPE max_value = FUNC_CALL(find_max_value)(partial_max, idx, input);
    
    UNIT_TYPE tmp_vals[ITEMS_NUM + 1];
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        tmp_vals[i] = native_exp(input[LWS * i + idx] - max_value);
    }
    tmp_vals[ITEMS_NUM] = idx < LEFTOVERS ? native_exp(input[LWS * ITEMS_NUM + idx] - max_value) : UNIT_VAL_ZERO;

    // accumulate all values;
    __local UNIT_TYPE partial_acc[LWS]; // all values accumulated;
    partial_acc[idx] = UNIT_VAL_ZERO;
    for(int i = 0; i < ITEMS_NUM + 1; i++)
    {
        partial_acc[idx] += tmp_vals[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE); // we must be sure that all threads calculated max of elements(we can remove it if simd32 and GWS <= 32
    if(idx == 0)
    {
        for(int i = 1; i < LWS; i++)
        {
            partial_acc[0] += partial_acc[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        output[LWS * i + idx] = tmp_vals[i] / partial_acc[0];
    }
    if(idx < LEFTOVERS)
        output[LWS * ITEMS_NUM + idx] = tmp_vals[ITEMS_NUM] / partial_acc[0];
}