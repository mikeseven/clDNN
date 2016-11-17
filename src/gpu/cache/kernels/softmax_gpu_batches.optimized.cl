float FUNC(find_max_value)(const int global_id, const int idx, const int batch_offset, const int batch_num, __global float* input)
{
    __local float partial_max[LWS];
    float value = -FLT_MAX;
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        value = max(value, input[LWS * i + global_id]);
    }
    value = max(value, global_id < LEFTOVERS? input[LWS * ITEMS_NUM + global_id] : -FLT_MAX);
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

KERNEL (softmax_gpu_batches)(__global neural_memory* input_mem, __global neural_memory* dst_mem)
{
    __global float* input = (__global float*)get_data(input_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);

    const int batch_num = INPUT_BATCH_NUM;
    const int global_id = get_global_id(0);
    const int idx = global_id / batch_num;

    const int batch_offset = global_id % batch_num;

    const float max_value = FUNC_CALL(find_max_value)(global_id, idx, batch_offset, batch_num, input);

    float tmp_vals[ITEMS_NUM + 1];
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        tmp_vals[i] = native_exp(input[LWS * i + global_id] - max_value);
    }
    tmp_vals[ITEMS_NUM] = global_id < LEFTOVERS ? native_exp(input[LWS * ITEMS_NUM + global_id] - max_value) : 0;

    // accumulate all values;
    __local float partial_acc[LWS]; // all values accumulated;
    partial_acc[global_id] = 0;
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
        pDst[LWS * i + global_id] = tmp_vals[i] / partial_acc[batch_offset];
    }
    if(global_id < LEFTOVERS)
        pDst[LWS * ITEMS_NUM + global_id] = tmp_vals[ITEMS_NUM] / partial_acc[batch_offset];
}