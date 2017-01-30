#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (batch_norm_use_global_stats_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* mean, __global UNIT_TYPE* variance)
{
	const uint feature_id = get_global_id(1);
	const uint linear_id = get_global_id(0) + feature_id * INPUT_BATCH_NUM + get_global_id(2) * INPUT_BATCH_NUM * INPUT_FEATURE_NUM;
	output[linear_id] = (input[linear_id] - mean[feature_id]) / (sqrt(variance[feature_id]) + EPSILON);
}