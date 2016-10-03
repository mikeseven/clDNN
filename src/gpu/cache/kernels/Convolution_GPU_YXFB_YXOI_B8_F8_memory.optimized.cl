#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

__attribute__((reqd_work_group_size(8, 1, 1))) 
KERNEL(Convolution_GPU_YXFB_YXOI_B8_F8_memory)(
    const __global float* input,
    __global float* output,
    const __global float* filter,
    const __global float* bias,
    uint split_idx)
{
	const uint batch_num = INPUT_BATCH_NUM;

	const uint linear_id_xy = get_global_id(1) + get_global_size(1) * get_global_id(2);
	const uint global_id = get_global_id(0) + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * FILTER_OUTPUT_FEATURE_NUM * batch_num;

	const uint out_batch_id = get_local_id(0);
	const uint out_fm = get_global_id(0) / INPUT_BATCH_NUM;
	const uint out_x = get_global_id(1);
	const uint out_y = get_global_id(2);

	const int ofm_offset = out_fm % FILTER_OUTPUT_FEATURE_NUM;

	float result = bias[ofm_offset];

	bool finish = false;

	finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
	finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

	float8 _data = 0.f;

	const uint sub_group_id = get_local_id(0);

	if(!finish)
	{
		const int x = out_x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
		const int y = out_y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

		for (uint i = 0; i < FILTER_SIZE_Y; i++)
		{
			const int input_offset_y = y + i;
			const bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;

			if(!zero_y)
			{
				for (uint j = 0; j < FILTER_SIZE_X; j++)
				{
					const int input_offset_x = x + j;
				
					const bool zero = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;

					if(!zero)
					{
						int input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * INPUT_FEATURE_NUM * batch_num;
						input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * batch_num;
						input_idx += out_batch_id;
				
						const int filter_idx = sub_group_id + FILTER_INPUT_FEATURE_NUM * ( ofm_offset +  FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));

						for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h+=8)
						{
							float f_val = filter[filter_idx + h];
							float8 _input = as_float8(intel_sub_group_block_read8((const __global uint*)input + input_idx + h * batch_num));
							_data.s0 = fma(_input.s0, intel_sub_group_shuffle(f_val, 0), _data.s0);
							_data.s1 = fma(_input.s1, intel_sub_group_shuffle(f_val, 1), _data.s1);
							_data.s2 = fma(_input.s2, intel_sub_group_shuffle(f_val, 2), _data.s2);
							_data.s3 = fma(_input.s3, intel_sub_group_shuffle(f_val, 3), _data.s3);
							_data.s4 = fma(_input.s4, intel_sub_group_shuffle(f_val, 4), _data.s4);
							_data.s5 = fma(_input.s5, intel_sub_group_shuffle(f_val, 5), _data.s5);
							_data.s6 = fma(_input.s6, intel_sub_group_shuffle(f_val, 6), _data.s6);
							_data.s7 = fma(_input.s7, intel_sub_group_shuffle(f_val, 7), _data.s7);
						}
					}
				} 
			}
		}
	}
	result += _data.s0 + _data.s1 + _data.s2 + _data.s3 +
			  _data.s4 + _data.s5 + _data.s6 + _data.s7;

	ACTIVATION(output[global_id], result);
}

#undef ACTIVATION