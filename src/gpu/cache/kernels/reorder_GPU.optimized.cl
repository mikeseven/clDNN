uint FUNC(OUT_FORMAT)(uint size[4], uint pos[4]) {
    OUT_FORMAT_IMPLEMENTATION
}
KERNEL (reorder_GPU)(__global neural_memory* in_mem, __global neural_memory* out_mem)
{
    __global float* input = (__global float*)get_data(in_mem);
    __global float* output = (__global float*)get_data(out_mem);

    uint pos1D = get_global_id(0);
    uint pos[DIMENSIONS]; // position in each of dimensions
	for(uint i = 0; i < DIMENSIONS; i++)
    {
		uint order_idx = CALCULATION_ORDER[i];
		pos[order_idx] = pos1D % SIZE[order_idx];
        pos1D /= SIZE[order_idx]; 
    }

    uint output_pos = FUNC_CALL(OUT_FORMAT)(SIZE, pos);

    output[output_pos] = input[get_global_id(0)];
}