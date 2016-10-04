#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL (Fully_Connected_GPU_yxfn_byxf_b8_f8_memory)(
    const __global float* input, 
    __global float* output, 
    const __global float* weight,
    const __global float* bias)
{
	const uint x = get_global_id(0);
	const int batch_id = x % INPUT_BATCH_NUM;
	uint neuronIdx = x / INPUT_BATCH_NUM;

	float result = bias[neuronIdx];

	float8 _data = 0.f;

	const uint sub_group_id = get_local_id(0);

	uint weight_offset = sub_group_id + neuronIdx * INPUT_FEATURE_NUM * INPUT_SIZE_Y * INPUT_SIZE_X;
	for(int j = 0; j < INPUT_SIZE_Y; j++)
	{
		for(int i = 0; i < INPUT_SIZE_X; i++)
		{    
			int input_idx = (i + j * INPUT_SIZE_X) * INPUT_FEATURE_NUM * INPUT_BATCH_NUM + batch_id;
			for(int k = 0; k < INPUT_FEATURE_NUM; k+=8)
			{
				const float weight_val = weight[weight_offset];
				const float8 _input = as_float8(intel_sub_group_block_read8((const __global uint*)input + input_idx + k * INPUT_BATCH_NUM));
				_data.s0 = fma(_input.s0, intel_sub_group_shuffle(weight_val, 0), _data.s0);
				_data.s1 = fma(_input.s1, intel_sub_group_shuffle(weight_val, 1), _data.s1);                                
				_data.s2 = fma(_input.s2, intel_sub_group_shuffle(weight_val, 2), _data.s2);
				_data.s3 = fma(_input.s3, intel_sub_group_shuffle(weight_val, 3), _data.s3);
				_data.s4 = fma(_input.s4, intel_sub_group_shuffle(weight_val, 4), _data.s4);
				_data.s5 = fma(_input.s5, intel_sub_group_shuffle(weight_val, 5), _data.s5);
				_data.s6 = fma(_input.s6, intel_sub_group_shuffle(weight_val, 6), _data.s6);
				_data.s7 = fma(_input.s7, intel_sub_group_shuffle(weight_val, 7), _data.s7);
				weight_offset += 8;
			}
		}
	}
	result += _data.s0 + _data.s1 + _data.s2 + _data.s3 +
			  _data.s4 + _data.s5 + _data.s6 + _data.s7;

	ACTIVATION(output[x], result);
}

#undef ACTIVATION