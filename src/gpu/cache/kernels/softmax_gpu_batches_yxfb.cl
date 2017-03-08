#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (softmax_gpu_batches_yxfb)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    
    const uint element_id = get_global_id(0); // flatten indexes in batch
    const uint batch_id = get_global_id(1); // index of batch
    const uint element_offset = INPUT_BATCH_NUM*INPUT_FEATURE_NUM;
    const uint feature_offset = INPUT_BATCH_NUM;
    const uint global_id = element_id*element_offset + batch_id;

    if (element_id >= ELEMENTS_NUM)
        return;

    UNIT_TYPE tmp_vals = UNIT_VAL_ZERO;
    UNIT_TYPE feature_maximum = -UNIT_VAL_MAX;
    UNIT_TYPE feature_sum = UNIT_VAL_ZERO;
    UNIT_TYPE vals[ITEMS_NUM];

    
    //find max and allocate inputs to vals
    for (uint i = 0; i<ITEMS_NUM; ++i)
    {
        tmp_vals = input[global_id + i*feature_offset];
        feature_maximum = max(feature_maximum, tmp_vals);
        vals[i] = tmp_vals;
    }

    //calculate native_exp and sum 
    for (uint i = 0; i<ITEMS_NUM; ++i)
    {
        tmp_vals = native_exp(vals[i] - feature_maximum);
        feature_sum += tmp_vals;
        vals[i] = tmp_vals;
    }


    for (uint i = 0; i<ITEMS_NUM; ++i)
    {
        output[global_id + i*feature_offset] = vals[i] / feature_sum;
    }
    
}