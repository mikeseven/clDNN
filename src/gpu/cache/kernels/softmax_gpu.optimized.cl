float FUNC(find_max_value)(const int idx, __global float* input)
{
    __local float partial_max[LWS];
    float value = -FLT_MAX;
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        value = max(value, input[LWS * i + idx]);
    }
    value = max(value, idx < LEFTOVERS? input[LWS * ITEMS_NUM + idx] : -FLT_MAX);
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

KERNEL (softmax_gpu)(__global float* input, __global float* pDst)
{
    const int idx = get_local_id(0);

    const float max_value = FUNC_CALL(find_max_value)(idx, input);
    
    float tmp_vals[ITEMS_NUM + 1];
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        tmp_vals[i] = native_exp(input[LWS * i + idx] - max_value);
    }
    tmp_vals[ITEMS_NUM] = idx < LEFTOVERS ? native_exp(input[LWS * ITEMS_NUM + idx] - max_value) : 0;

    // accumulate all values;
    __local float partial_acc[LWS]; // all values accumulated;
    partial_acc[idx] = 0;
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
        pDst[LWS * i + idx] = tmp_vals[i] / partial_acc[0];
    }
    if(idx < LEFTOVERS)
        pDst[LWS * ITEMS_NUM + idx] = tmp_vals[ITEMS_NUM] / partial_acc[0];
}