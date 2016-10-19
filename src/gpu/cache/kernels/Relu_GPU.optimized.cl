#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

KERNEL(Relu_GPU)(const __global neural_memory* input_mem, __global neural_memory* output_mem)
{
    const __global float* input = (const __global float*)get_data(input_mem);
    __global float* output = (__global float*)get_data(output_mem);
    
    const int global_id = get_global_id(0);
    ACTIVATION(output[global_id], input[global_id]);
}

#undef ACTIVATION