#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (batch_norm_use_global_stats_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* mean, __global UNIT_TYPE* variance)
{
    const uint element_offset = get_global_id(1) * INPUT_BATCH_NUM + get_global_id(2) * INPUT_BATCH_NUM * INPUT_FEATURE_NUM;
	const uint linear_id = get_global_id(0) + element_offset;
	
#if BFYX_MEAN_FORMAT_USED
	// BFYX format of mean
	const uint mean_offset = (get_global_id(2) % INPUT_SIZE_X + MEAN_SIZE_X) * ((get_global_id(2) / INPUT_SIZE_Y) + MEAN_SIZE_Y * (get_global_id(1) + MEAN_FEATURE_NUM * (get_global_id(0) % MEAN_BATCH_NUM)));
#else
    // YXFB format of mean
	const uint mean_offset = element_offset / INPUT_BATCH_NUM;
#endif

#if BFYX_VARIANCE_FORMAT_USED
	// BFYX format of variance
	const uint var_offset = (get_global_id(2) % INPUT_SIZE_X + VARIANCE_SIZE_X) * ((get_global_id(2) / INPUT_SIZE_Y) + VARIANCE_SIZE_Y * (get_global_id(1) + VARIANCE_FEATURE_NUM * (get_global_id(0) % VARIANCE_BATCH_NUM)));
#else
    // YXFB format of variance
	const uint var_offset = element_offset / INPUT_BATCH_NUM;
#endif

	output[linear_id] = (input[linear_id] - mean[mean_offset]) / sqrt(variance[var_offset] + EPSILON);
}