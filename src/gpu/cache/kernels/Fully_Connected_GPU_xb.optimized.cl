#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

KERNEL (Fully_Connected_GPU_xb)(
    const __global float* input,
    __global float* output)
{
	const int x = get_global_id(0);

	uint outXIdx = x / INPUT_BATCH_NUM;
	uint inputBatchIdx = x % INPUT_BATCH_NUM;
	uint weightBatchIdx = outXIdx * WEIGHTS_BATCH_NUM;
	float result = BIASES[outXIdx];
	for (uint i = 0; i < INPUT_SIZE_X; i++)
	{
		result += input[i * INPUT_BATCH_NUM + inputBatchIdx] * WEIGHTS[weightBatchIdx + i];
	}
	ACTIVATION(output[x], result);
}

#undef ACTIVATION