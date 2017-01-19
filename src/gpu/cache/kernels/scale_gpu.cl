#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (scale_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* scale_input)
{
	const uint linear_id = get_global_id(0) + get_global_id(1) * INPUT_BATCH_NUM + get_global_id(2) * INPUT_BATCH_NUM * INPUT_FEATURE_NUM;
	output[linear_id] = input[linear_id] * scale_input[linear_id];
}