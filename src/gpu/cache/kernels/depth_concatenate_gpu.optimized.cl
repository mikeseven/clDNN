KERNEL (depth_concatenate_gpu)(__global float* input, __global float* output, uint depth_offset)
{
    uint global_id = get_global_id(0);

    uint input_offset = global_id * INPUT_FEATURE_NUM * OUTPUT_BATCH_NUM;
    uint output_offset = OUTPUT_BATCH_NUM * (depth_offset + global_id * OUTPUT_FEATURE_NUM);
    for(uint f = 0; f < INPUT_FEATURE_NUM * OUTPUT_BATCH_NUM; f++)
    {
        output[output_offset++] = input[input_offset++];
    }
}