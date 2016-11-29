#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

KERNEL (fully_connected_gpu_xb_xb_b16)(
    const __global float* input, 
    __global float* output, 
    const __global float* weights,
    const __global float* bias)
{
	const uint global_id = get_global_id(0);
	const uint local_id = get_local_id(0);
	const uint batch_id = local_id + get_local_size(0) * BATCHES_PER_WORK_ITEM * (get_group_id(0) % LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS);

	uint neuronIdx = (global_id / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS) * NEURONS_PER_WORK_ITEM;

	const int out_id = neuronIdx * INPUT_BATCH_NUM + batch_id;

	float8 _data[BATCHES_PER_WORK_ITEM];
	for(uint i = 0; i < BATCHES_PER_WORK_ITEM; i++)
	{
		_data[i] = 0.f;
	}

	uint weight_offset = local_id + neuronIdx;
	uint input_idx = batch_id;

	for(uint h = 0; h < INPUT_ELEMENTS_COUNT; h++)
	{
#if BATCHES_PER_WORK_ITEM == 2
		float2 _input = as_float2(intel_sub_group_block_read2((const __global uint*)input + input_idx));
		DOT_PRODUCT_8(_data[0], _input.s0, weights[weight_offset])
		DOT_PRODUCT_8(_data[1], _input.s1, weights[weight_offset])
		input_idx += INPUT_BATCH_NUM;
#else
		for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
		{
			DOT_PRODUCT_8(_data[s], input[input_idx], weights[weight_offset])
			input_idx += LOCAL_WORK_GROUP_SIZE;
		}
		input_idx += INPUT_BATCH_NUM - BATCHES_PER_WORK_ITEM * LOCAL_WORK_GROUP_SIZE;
#endif
		weight_offset+= WEIGHTS_BATCH_NUM;
	}


	for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
	{
		float bias_val = bias[neuronIdx + local_id];
		ADD_BIAS_8(_data[s], bias_val);
	}

	for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
	{
		ACTIVATION(_data[s], _data[s]);
	}

	for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
	{
		int _out_id = out_id + s * LOCAL_WORK_GROUP_SIZE;
		output[_out_id] = _data[s].s0; _out_id += INPUT_BATCH_NUM;
		output[_out_id] = _data[s].s1; _out_id += INPUT_BATCH_NUM;
		output[_out_id] = _data[s].s2; _out_id += INPUT_BATCH_NUM;
		output[_out_id] = _data[s].s3; _out_id += INPUT_BATCH_NUM;
		output[_out_id] = _data[s].s4; _out_id += INPUT_BATCH_NUM;
		output[_out_id] = _data[s].s5; _out_id += INPUT_BATCH_NUM;
		output[_out_id] = _data[s].s6; _out_id += INPUT_BATCH_NUM;
		output[_out_id] = _data[s].s7; _out_id += INPUT_BATCH_NUM;
	}
}

#undef ACTIVATION