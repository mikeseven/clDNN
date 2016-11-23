#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

KERNEL (fully_connected_gpu_xb_bx)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
	const int x = get_global_id(0);
	const uint batch_id = x % INPUT_BATCH_NUM;

	uint outXIdx = x / INPUT_BATCH_NUM;
	uint weight_offset = outXIdx * INPUT_ELEMENTS_COUNT;
	float result = bias[outXIdx];
	for (uint i = 0; i < INPUT_ELEMENTS_COUNT; i++)
	{
		result += input[i * INPUT_BATCH_NUM + batch_id] * weight[weight_offset++];
	}
	ACTIVATION(output[x], result);
}

#undef ACTIVATION