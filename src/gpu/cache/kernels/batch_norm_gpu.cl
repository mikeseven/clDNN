#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (batch_norm_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* mean, __global UNIT_TYPE* variance)
{
    const uint element_offset = get_global_id(1) * INPUT_BATCH_NUM + get_global_id(2) * INPUT_BATCH_NUM * INPUT_FEATURE_NUM;
	const uint linear_id = get_global_id(0) + element_offset;
	const uint mean_var_offset = element_offset / INPUT_BATCH_NUM;
	
	//compute mean
	UNIT_TYPE acc = UNIT_VAL_ZERO;
	for(int i = 0; i < INPUT_BATCH_NUM; i++)
	{
		acc += input[element_offset + i];
	}
	
	mean[mean_var_offset] = acc / INPUT_BATCH_NUM;
	UNIT_TYPE mean_subtract = input[linear_id] - mean[mean_var_offset];

	//compute variance using var(X) = E((X-EX)^2)
	acc = UNIT_VAL_ZERO;	
	for(int i = 0; i < INPUT_BATCH_NUM; i++)
	{
		acc += native_powr(mean_subtract, 2);
	}
	variance[mean_var_offset] = acc / INPUT_BATCH_NUM;
	
	output[linear_id] = mean_subtract / sqrt(variance[mean_var_offset] + EPSILON);
}