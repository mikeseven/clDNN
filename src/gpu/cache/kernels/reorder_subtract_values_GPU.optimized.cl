uint FUNC(OUT_FORMAT)(uint size[DIMENSIONS], uint pos[DIMENSIONS]) {
    OUT_FORMAT_IMPLEMENTATION
}
KERNEL (reorder_subtract_values_GPU)(__global float* input, __global float* output)
{
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
    uint input_idx = (global_id_2 * global_size_1 + global_id_1) * global_size_0 + global_id_0;
    output[output_pos] = input[input_idx] - VALUE_TO_SUBTRACT[pos[1]];
}