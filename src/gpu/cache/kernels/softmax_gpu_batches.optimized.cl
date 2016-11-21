#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif



UNIT_TYPE FUNC(find_max_value)(__local UNIT_TYPE* partial_max, const int global_id, const int idx, const int batch_offset, const int batch_num, const __global UNIT_TYPE* input)
{
    UNIT_TYPE value = -UNIT_VAL_MAX;
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        value = max(value, input[LWS * i + global_id]);
    }
    value = max(value, global_id < LEFTOVERS? input[LWS * ITEMS_NUM + global_id] : -UNIT_VAL_MAX);
    partial_max[global_id] = value;

    barrier(CLK_LOCAL_MEM_FENCE);
    if(global_id < batch_num)
    {
        for(int i = 1; i < LWS / batch_num; i++)
        {
            partial_max[batch_offset] = max(partial_max[0], partial_max[i*batch_num + batch_offset]);
        };
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return partial_max[batch_offset];
}

KERNEL (softmax_gpu_batches)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const int batch_num = INPUT_BATCH_NUM;
    const int global_id = get_global_id(0);
    const int idx = global_id / batch_num;

    const int batch_offset = global_id % batch_num;

    __local UNIT_TYPE partial_max[LWS];
    const UNIT_TYPE max_value = FUNC_CALL(find_max_value)(partial_max, global_id, idx, batch_offset, batch_num, input);

    UNIT_TYPE tmp_vals[ITEMS_NUM + 1];
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        tmp_vals[i] = native_exp(input[LWS * i + global_id] - max_value);
    }
    tmp_vals[ITEMS_NUM] = global_id < LEFTOVERS ? native_exp(input[LWS * ITEMS_NUM + global_id] - max_value) : UNIT_VAL_ZERO;

    // accumulate all values;
    __local UNIT_TYPE partial_acc[LWS]; // all values accumulated;
    partial_acc[global_id] = UNIT_VAL_ZERO;
    for(int i = 0; i < ITEMS_NUM + 1; i++)
    {
        partial_acc[global_id] += tmp_vals[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE); // we must be sure that all threads calculated max of elements(we can remove it if simd32 and GWS <= 32
    if(global_id < batch_num)
    {
        for(int i = 1; i < LWS/batch_num; i++)
        {
            partial_acc[batch_offset] += partial_acc[i*batch_num + batch_offset];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        output[LWS * i + global_id] = tmp_vals[i] / partial_acc[batch_offset];
    }
    if(global_id < LEFTOVERS)
        output[LWS * ITEMS_NUM + global_id] = tmp_vals[ITEMS_NUM] / partial_acc[batch_offset];
}