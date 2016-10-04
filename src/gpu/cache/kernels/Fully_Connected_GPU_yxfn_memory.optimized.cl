#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

KERNEL (Fully_Connected_GPU_yxfn_memory)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
	const uint x = get_global_id(0);
	const int batch_id = x % INPUT_BATCH_NUM;
	uint neuronIdx = x / INPUT_BATCH_NUM;

	float result = bias[neuronIdx];

	uint weight_offset = neuronIdx * INPUT_FEATURE_NUM * INPUT_SIZE_Y * INPUT_SIZE_X;
	for(int k = 0; k < INPUT_FEATURE_NUM; k++)
		for(int j = 0; j < INPUT_SIZE_Y; j++)
			for(int i = 0; i < INPUT_SIZE_X; i++)
			{
				result += input[(k + INPUT_FEATURE_NUM * ( i + j * INPUT_SIZE_X)) * INPUT_BATCH_NUM + batch_id] * weight[weight_offset++];
			}
	ACTIVATION(output[x], result);
}

#undef ACTIVATION