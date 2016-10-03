#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

KERNEL(Convolution_GPU_YXFB)(
    const __global float* input,
    __global float* output)
{
	const uint global_id = get_global_id(0);
	const uint batch_num = OUTPUT_BATCH_NUM;
	const uint batch_offset = global_id % batch_num;

	const uint linear_id = get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2));
	const uint ofm_offset = (global_id / batch_num) % (OUTPUT_FEATURE_NUM / FILTER_ARRAY_NUM);

	const uint f_ofm_offset = ofm_offset * FILTER_SIZE_Y * FILTER_SIZE_X * FILTER_INPUT_FEATURE_NUM;

	const int idx = (global_id / batch_num) / FILTER_ARRAY_NUM;

	const int i_ifm_num = INPUT_FEATURE_NUM;

	const uint out_x = get_global_id(1);
	const uint out_y = get_global_id(2);

	const int x = out_x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
	const int y = out_y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

	const int split_idx = ((global_id / batch_num) / FILTER_OUTPUT_FEATURE_NUM) % FILTER_ARRAY_NUM;
	float result = BIAS[split_idx][ofm_offset];

	bool finish = false;

	finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
	finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

	if(!finish)
	{
		for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
		{
			const int f_ifm_offset = h * FILTER_SIZE_Y * FILTER_SIZE_X;
			for (uint i = 0; i < FILTER_SIZE_Y; i++)
			{
				for (uint j = 0; j < FILTER_SIZE_X; j++)
				{
					int input_offset_x = x + j;
					int input_offset_y = y + i;

					bool zero = false;
					zero = input_offset_x < 0 ? true : zero;
					zero = input_offset_y < 0 ? true : zero;
					zero = input_offset_x >= INPUT_SIZE_X ? true : zero;
					zero = input_offset_y >= INPUT_SIZE_Y ? true : zero;

					int input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * i_ifm_num * batch_num;
					input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * batch_num;
					input_idx += h * batch_num;
					input_idx += batch_offset;
					int filter_idx = (i * FILTER_SIZE_X + j) + f_ofm_offset + f_ifm_offset;
					result += zero ? 0 : input[input_idx] * FILTER[split_idx][filter_idx];
				}
			}
		}
	}
        
    ACTIVATION(output[linear_id], result);
}

#undef ACTIVATION