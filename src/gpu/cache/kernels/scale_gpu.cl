#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (scale_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* scale_input
#if BIAS_TERM
, __global UNIT_TYPE* bias)
#else
)
#endif
{
	const uint linear_id = get_global_id(0) + INPUT_BATCH_NUM * (get_global_id(1) + get_global_id(2) * INPUT_FEATURE_NUM);
	const uint scale_batch_id = (SCALE_BATCH_NUM == 1) ? 0 : get_global_id(0);
	const uint scale_feature_id = (SCALE_FEATURE_NUM == 1) ? 0 : get_global_id(1);
	const uint x = (SCALE_SIZE_X == 1) ? 0 : ((SCALE_SIZE_Y == 1) ? (get_global_id(2) / INPUT_SIZE_Y) : (get_global_id(2) % SCALE_SIZE_X));
	const uint y = (SCALE_SIZE_Y == 1) ? 0 : ((SCALE_SIZE_X == 1) ? (get_global_id(2) % SCALE_SIZE_Y) : (get_global_id(2) / SCALE_SIZE_X));
	const uint scale_linear_id = scale_batch_id + SCALE_BATCH_NUM * (scale_feature_id + SCALE_FEATURE_NUM * (x + y * SCALE_SIZE_X));

	#if BIAS_TERM
	output[linear_id] = mad(input[linear_id], scale_input[scale_linear_id], bias[scale_linear_id]);
	#else
	output[linear_id] = input[linear_id] * scale_input[scale_linear_id];
	#endif
}