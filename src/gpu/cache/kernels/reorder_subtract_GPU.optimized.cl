uint FUNC(OUT_FORMAT)(uint size[DIMENSIONS], uint pos[DIMENSIONS]) {
    OUT_FORMAT_IMPLEMENTATION
}
uint FUNC(SUBTRACT_FORMAT)(uint size[DIMENSIONS], uint pos[DIMENSIONS]) {

	SUBTRACT_FORMAT_IMPLEMENTATION
}
KERNEL (reorder_subtract_GPU)(__global neural_memory* in_mem, __global neural_memory* out_mem, __global neural_memory* subtract_values)
{
    __global float* input = (__global float*)get_data(in_mem);
    __global float* output = (__global float*)get_data(out_mem);
    __global float* subtract = (__global float*)get_data(subtract_values);

    const uint global_id_0 = get_global_id(0);
    const uint global_id_1 = get_global_id(1);
    const uint global_id_2 = get_global_id(2);
    const uint global_size_1 = get_global_size(1);
    const uint global_size_0 = get_global_size(0);

    uint pos[DIMENSIONS]; // position in each of dimensions
    pos[CALCULATION_ORDER[DIMENSIONS-1]] = global_id_2;
    pos[CALCULATION_ORDER[DIMENSIONS-2]] = global_id_1;
    uint pos1D = global_id_0;
	for(uint i = 0; i < DIMENSIONS-2; i++)
    {
		uint order_idx = CALCULATION_ORDER[i];
		pos[order_idx] = pos1D % SIZE[order_idx];
        pos1D /= SIZE[order_idx];
    }

    uint output_pos = FUNC_CALL(OUT_FORMAT)(SIZE, pos);
	// We set it to 0 because we subtract the same values from every input batch
	pos[0] = 0;
	uint subtract_pos = FUNC_CALL(SUBTRACT_FORMAT)(SIZE, pos);
    uint input_idx = (global_id_2 * global_size_1 + global_id_1) * global_size_0 + global_id_0;
    output[output_pos] = input[input_idx] - subtract[subtract_pos];
}