#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

KERNEL (Fully_Connected_GPU_xb_xb_memory)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
	const uint x = get_global_id(0);
	const uint batch_id = x % INPUT_BATCH_NUM;

	const uint outXIdx = x / INPUT_BATCH_NUM;
	float result = 0;
	
	uint input_idx = batch_id;
	uint weight_idx = outXIdx;
	for (uint i = 0; i < INPUT_ELEMENTS_COUNT; i++)
	{
		result += input[input_idx] * weight[weight_idx];
		input_idx += INPUT_BATCH_NUM;
		weight_idx += WEIGHTS_BATCH_NUM;
	}

	result += bias[outXIdx];
	ACTIVATION(output[x], result);
}

#undef ACTIVATION