#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

__attribute__((reqd_work_group_size(8, 1, 1))) 
KERNEL(Convolution_GPU_YXFB_YXOI_B8_memory)(
    const __global float* input,
    __global float* output,
    const __global float* filter,
    const __global float* bias,
    uint split_idx)
{
	const int batch_num = INPUT_BATCH_NUM;

	const uint linear_id_xy = get_global_id(1) + get_global_size(1) * get_global_id(2);
	// we're computing 8 OUTPUT_FEATURE_MAP so we must divide by 8, but we got 8 batches, so no division is needed.
	int global_id = (get_global_id(0) / batch_num) * 8 + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * (FILTER_OUTPUT_FEATURE_NUM / OFM_PER_WORK_ITEM) * batch_num; 

	const uint out_batch_id = get_local_id(0);
	const uint out_x = get_global_id(1);
	const uint out_y = get_global_id(2);

	const int out_id = (global_id / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;

	const int ofm_offset = (global_id * (OFM_PER_WORK_ITEM / batch_num)) % FILTER_OUTPUT_FEATURE_NUM;

	bool finish = false;

	finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
	finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;

	const uint sub_group_id = get_local_id(0);

	float8 _data0 = 0.f;
	float8 _data1 = 0.f;

	if(!finish)
	{
		const int x = out_x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
		const int y = out_y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

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
						input_idx += out_batch_id;
					
						//sub_group_id used as offset to make each workitem load different filter, and then shuffle it
						int filter_idx = FILTER_INPUT_FEATURE_NUM * ( ofm_offset + sub_group_id +  FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));

						for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
						{
							DOT_PRODUCT_8(_data0, input[input_idx + h * batch_num], filter[filter_idx + h])
							DOT_PRODUCT_8(_data1, input[input_idx + h * batch_num], filter[filter_idx + h + FILTER_INPUT_FEATURE_NUM * 8])
						}
					}
				} 
			}
		}
	}

	ADD_BIAS_8(_data0, bias[ofm_offset + sub_group_id]);
	ADD_BIAS_8(_data1, bias[ofm_offset + sub_group_id + 8]);

	ACTIVATION_8(_data0);
	ACTIVATION_8(_data1);

	intel_sub_group_block_write8((__global uint*)output + out_id, as_uint8(_data0));
	intel_sub_group_block_write8((__global uint*)output + out_id + 8 * batch_num, as_uint8(_data1));
}

#undef ACTIVATION