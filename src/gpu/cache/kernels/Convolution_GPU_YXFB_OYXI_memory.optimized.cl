#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

KERNEL(Convolution_GPU_YXFB_OYXI_memory)(
    const __global float* input,
    __global float* output,
    const __global float* filter,
    const __global float* bias,
    uint split_idx)
{
	const int batch_num = INPUT_BATCH_NUM;

	const int bifn_num = batch_num * FILTER_OUTPUT_FEATURE_NUM;
	int global_id = get_global_id(0) % bifn_num + (get_global_id(0) / bifn_num) * bifn_num * FILTER_ARRAY_NUM + split_idx * bifn_num;

	const int ofm_offset = (global_id / batch_num) % (OUTPUT_FEATURE_NUM / FILTER_ARRAY_NUM);

	float result = bias[ofm_offset];

	bool finish = false;
	const uint out_x = global_id % OUTPUT_SIZE_X;
	const uint out_y = (global_id % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y)) / OUTPUT_SIZE_X;

	finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
	finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

	if(!finish)
	{
		const int batch_offset = global_id % batch_num;

		const int idx = (global_id / batch_num) / FILTER_ARRAY_NUM;

		const int x = ((idx / FILTER_OUTPUT_FEATURE_NUM) % OUTPUT_SIZE_X) * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
		const int y = ((idx / FILTER_OUTPUT_FEATURE_NUM) / OUTPUT_SIZE_X * STRIDE_SIZE_Y) + INPUT_OFFSET_SIZE_Y;

		const int f_ofm_offset = ofm_offset * FILTER_INPUT_FEATURE_NUM * FILTER_SIZE_X * FILTER_SIZE_Y;
		for (uint i = 0; i < FILTER_SIZE_Y; i++)
		{
			int input_offset_y = y + i;
			bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;

			if(!zero_y)
			{
				for (uint j = 0; j < FILTER_SIZE_X; j++)
				{
					int input_offset_x = x + j;
				
					bool zero = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;

					if(!zero)
					{
						int input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * INPUT_FEATURE_NUM * batch_num;
						input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * batch_num;
						input_idx += batch_offset;
				
						int filter_idx = f_ofm_offset + FILTER_INPUT_FEATURE_NUM * ( i * FILTER_SIZE_X + j);

						for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
						{
							result += input[input_idx + h * batch_num] * filter[filter_idx + h];
						}
					}
				} 
			}
		}
	}
	ACTIVATION(output[global_id], result);
}

#undef ACTIVATION