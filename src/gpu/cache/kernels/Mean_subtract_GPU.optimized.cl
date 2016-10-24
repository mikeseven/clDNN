KERNEL(Mean_subtract_GPU)(const __global neural_memory* input_mem, __global neural_memory* output_mem, const __global neural_memory* mean_mem)
{
    const __global float* input = (const __global float*)get_data(input_mem);
    __global float* output = (__global float*)get_data(output_mem);
    const __global float* mean = (const __global float*)get_data(mean_mem);

    const uint batch_num = INPUT_BATCH_NUM;

    const int global_id = get_global_id(0);
    const int batch_id = global_id % batch_num;
    const int feature_id = (global_id / batch_num) % INPUT_FEATURE_NUM;
    const int x = ((global_id / batch_num) / INPUT_FEATURE_NUM) % INPUT_SIZE_X;
    const int y = ((global_id / batch_num) / INPUT_FEATURE_NUM) / INPUT_SIZE_X;
    output[global_id] = input[global_id] - mean[x + MEAN_SIZE_X * (y + MEAN_SIZE_Y * (feature_id + MEAN_FEATURE_NUM * (batch_id % MEAN_BATCH_NUM)))];
}