#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

KERNEL(Convolution_GPU_BFXY)(
    const __global float* input,
    __global float* output)
{
	const int global_id = get_global_id(0);

	const int output_feature_num = OUTPUT_FEATURE_NUM;
	const int output_feature_size = OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
	const int output_batch_size = output_feature_num * output_feature_size;

	const int output_feature_idx = (global_id / output_feature_size ) % output_feature_num;
	const int batch_idx = global_id / output_batch_size;

	const int filter_input_feature_size = FILTER_SIZE_X * FILTER_SIZE_Y;

	const int filter_output_feature_num = FILTER_OUTPUT_FEATURE_NUM;
	const int filter_output_feature_size = FILTER_INPUT_FEATURE_NUM * filter_input_feature_size;
	const int filter_output_feature_offset = output_feature_idx * filter_output_feature_size;

	const int input_feature_num = INPUT_FEATURE_NUM;
	const int input_feature_size = INPUT_SIZE_X * INPUT_SIZE_Y;

	const int input_batch_size = input_feature_num * input_feature_size;
	const int input_batch_offset = input_batch_size * batch_idx;

	const int input_x_offset = global_id % (INPUT_SIZE_X / STRIDE_SIZE_X) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;    
	const int input_y_offset = ((global_id / (INPUT_SIZE_X / STRIDE_SIZE_X)) % (INPUT_SIZE_Y / STRIDE_SIZE_Y)) * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

	const int input_offset = input_batch_offset + input_y_offset * INPUT_SIZE_X + input_x_offset;

	// TODO!!!! change [0] from BIAS and FILTER to something that works - [0] is for temporary compilation
	float result = BIAS[0][output_feature_idx];

	for(uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
	{
		const int filter_input_feature_offset = h * filter_input_feature_size;   
		const int input_feature_offset = h * input_feature_size;
		for( uint i = 0; i < FILTER_SIZE_Y; i++)
		{
			for (uint j = 0; j < FILTER_SIZE_X; j++)
			{
				int input_idx = j + i * INPUT_SIZE_X + input_offset + input_feature_offset;
				int filter_idx = (i * FILTER_SIZE_X + j) + filter_output_feature_offset + filter_input_feature_offset;
				result += input[input_idx] * FILTER[0][filter_idx];
			}
		}
	}
	ACTIVATION(output[global_id], result);
}

#undef ACTIVATION