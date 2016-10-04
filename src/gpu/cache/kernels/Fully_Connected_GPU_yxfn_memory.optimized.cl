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
	const uint global_id = get_global_id(0);
	const uint group_id = get_global_id(1); // which part of batches we are computing, for example for batch 64 we compute batches 0..31 for group_id == 0 and batches 32..65 for group_id == 1
	uint sub_group_idx = get_local_id(0) % 8;

	const uint out_id = (sub_group_idx * BATCHES_PER_WORK_ITEM * get_global_size(1)) / 8 + (global_id / 8) * BATCHES_PER_WORK_ITEM * NEURONS_PER_WORK_ITEM * get_global_size(1) + (BATCHES_PER_WORK_ITEM * group_id) / 8;

	uint neuronIdx = sub_group_idx + (global_id / 8) * 8 * NEURONS_PER_WORK_ITEM;

	float8 blockC00 = 0.f;
	float8 blockC10 = 0.f;

#if BATCHES_PER_WORK_ITEM >= 16
	float8 blockC01 = 0.f;
	float8 blockC11 = 0.f;
#endif

#if BATCHES_PER_WORK_ITEM >= 32
	float8 blockC02 = 0.f;
	float8 blockC12 = 0.f;

	float8 blockC03 = 0.f;
	float8 blockC13 = 0.f;
#endif

	uint weight_offset = neuronIdx;
#if NEURONS_PER_WORK_ITEM > 1

	uint weight_offset2 = neuronIdx + 8;

#endif // #if NEURONS_PER_WORK_ITEM > 1

	uint input_idx = sub_group_idx * (BATCHES_PER_WORK_ITEM / 8) * get_global_size(1) + (group_id * BATCHES_PER_WORK_ITEM) / 8;
	for(uint h = 0; h < INPUT_ELEMENTS_COUNT / 8; h++)
	{
		float8 blockA00 = vload8(input_idx, input);

#if BATCHES_PER_WORK_ITEM >= 16
		float8 blockA01 = vload8(input_idx + 1, input);
#endif

#if BATCHES_PER_WORK_ITEM >= 32
		float8 blockA02 = vload8(input_idx + 2, input);
		float8 blockA03 = vload8(input_idx + 3, input);
#endif
		float8 blockB00;
		blockB00.s0 = weight[weight_offset]; weight_offset += WEIGHTS_BATCH_NUM;
		blockB00.s1 = weight[weight_offset]; weight_offset += WEIGHTS_BATCH_NUM;
		blockB00.s2 = weight[weight_offset]; weight_offset += WEIGHTS_BATCH_NUM;
		blockB00.s3 = weight[weight_offset]; weight_offset += WEIGHTS_BATCH_NUM;
		blockB00.s4 = weight[weight_offset]; weight_offset += WEIGHTS_BATCH_NUM;
		blockB00.s5 = weight[weight_offset]; weight_offset += WEIGHTS_BATCH_NUM;
		blockB00.s6 = weight[weight_offset]; weight_offset += WEIGHTS_BATCH_NUM;
		blockB00.s7 = weight[weight_offset]; weight_offset += WEIGHTS_BATCH_NUM;
		MULTIPLY_BLOCKS_8x8(blockC00, blockA00, blockB00)

#if BATCHES_PER_WORK_ITEM >= 16
		MULTIPLY_BLOCKS_8x8(blockC01, blockA01, blockB00)
#endif

#if BATCHES_PER_WORK_ITEM >= 32
		MULTIPLY_BLOCKS_8x8(blockC02, blockA02, blockB00)
		MULTIPLY_BLOCKS_8x8(blockC03, blockA03, blockB00)
#endif

#if NEURONS_PER_WORK_ITEM > 1

		float8 blockB10;
		blockB10.s0 = weight[weight_offset2]; weight_offset2 += WEIGHTS_BATCH_NUM;
		blockB10.s1 = weight[weight_offset2]; weight_offset2 += WEIGHTS_BATCH_NUM;
		blockB10.s2 = weight[weight_offset2]; weight_offset2 += WEIGHTS_BATCH_NUM;
		blockB10.s3 = weight[weight_offset2]; weight_offset2 += WEIGHTS_BATCH_NUM;
		blockB10.s4 = weight[weight_offset2]; weight_offset2 += WEIGHTS_BATCH_NUM;
		blockB10.s5 = weight[weight_offset2]; weight_offset2 += WEIGHTS_BATCH_NUM;
		blockB10.s6 = weight[weight_offset2]; weight_offset2 += WEIGHTS_BATCH_NUM;
		blockB10.s7 = weight[weight_offset2]; weight_offset2 += WEIGHTS_BATCH_NUM;
		MULTIPLY_BLOCKS_8x8(blockC10, blockA00, blockB10)

#if BATCHES_PER_WORK_ITEM >= 16
		MULTIPLY_BLOCKS_8x8(blockC11, blockA01, blockB10)
#endif
#if BATCHES_PER_WORK_ITEM >= 32
		MULTIPLY_BLOCKS_8x8(blockC12, blockA02, blockB10)
		MULTIPLY_BLOCKS_8x8(blockC13, blockA03, blockB10)
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1
		input_idx += INPUT_BATCH_NUM; // we don't need to multiply by 8 because of vload8
	}

    blockC00 += bias[neuronIdx];
#if BATCHES_PER_WORK_ITEM >= 16
    blockC01 += bias[neuronIdx];
#endif

#if BATCHES_PER_WORK_ITEM >= 32
    blockC02 += bias[neuronIdx];
    blockC03 += bias[neuronIdx];
#endif

#if NEURONS_PER_WORK_ITEM > 1

    blockC10 += bias[neuronIdx+8];
#if BATCHES_PER_WORK_ITEM >= 16
    blockC11 += bias[neuronIdx+8];
#endif
#if BATCHES_PER_WORK_ITEM >= 32
    blockC12 += bias[neuronIdx+8];
    blockC13 += bias[neuronIdx+8];
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1

    ACTIVATION_8(blockC00);
#if BATCHES_PER_WORK_ITEM >= 16
    ACTIVATION_8(blockC01);
#endif
#if BATCHES_PER_WORK_ITEM >= 32
    ACTIVATION_8(blockC02);
    ACTIVATION_8(blockC03);
#endif

#if NEURONS_PER_WORK_ITEM > 1

    ACTIVATION_8(blockC10);
#if BATCHES_PER_WORK_ITEM >= 16
    ACTIVATION_8(blockC11);
#endif
#if BATCHES_PER_WORK_ITEM >= 32
    ACTIVATION_8(blockC12);
    ACTIVATION_8(blockC13);
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1

    vstore8(blockC00, out_id, output);
#if BATCHES_PER_WORK_ITEM >= 16
    vstore8(blockC01, out_id + 1, output);
#endif
#if BATCHES_PER_WORK_ITEM >= 32
    vstore8(blockC02, out_id + 2, output);
    vstore8(blockC03, out_id + 3, output);
#endif

#if NEURONS_PER_WORK_ITEM > 1

    vstore8(blockC10, out_id+INPUT_BATCH_NUM, output);

#if BATCHES_PER_WORK_ITEM >= 16
    vstore8(blockC11, out_id+INPUT_BATCH_NUM+1, output);
#endif

#if BATCHES_PER_WORK_ITEM >= 32
    vstore8(blockC12, out_id+INPUT_BATCH_NUM+2, output);
    vstore8(blockC13, out_id+INPUT_BATCH_NUM+3, output);
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1
}

#undef ACTIVATION