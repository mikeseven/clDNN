#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (batch_norm_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* mean, __global UNIT_TYPE* variance)
{
	const uint feature_id = get_global_id(1);
	const uint feature_offset = feature_id * INPUT_BATCH_NUM;
	const uint linear_id = get_global_id(0) + feature_offset + get_global_id(2) * INPUT_BATCH_NUM * INPUT_FEATURE_NUM;
	
	//compute mean
	UNIT_TYPE acc = UNIT_VAL_ZERO;
	for(int i = 0; i < INPUT_BATCH_NUM; i++)
	{
		for(int j = 0; j < INPUT_SIZE_X * INPUT_SIZE_Y; j++)
		{
		acc += input[feature_offset + i + j * INPUT_BATCH_NUM * INPUT_FEATURE_NUM];
		}
	}
	mean[feature_id] = acc / (INPUT_BATCH_NUM * INPUT_SIZE_X * INPUT_SIZE_Y);
	output[linear_id] = input[linear_id] - mean[feature_id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	//compute variance using var(X) = E((X-EX)^2)
	acc = UNIT_VAL_ZERO;	
	for(int i = 0; i < INPUT_BATCH_NUM; i++)
	{
		for(int j = 0; j < INPUT_SIZE_X * INPUT_SIZE_Y; j++)
		{
		acc += native_powr(output[feature_offset + i + j * INPUT_BATCH_NUM * INPUT_FEATURE_NUM], 2);
		}
	}
	variance[feature_id] = acc / (INPUT_BATCH_NUM * INPUT_SIZE_X * INPUT_SIZE_Y);
	
	output[linear_id] = output[linear_id] / (sqrt(variance[feature_id]) + EPSILON);
}