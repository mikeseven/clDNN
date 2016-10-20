KERNEL (lrn_GPU)(__global float* input, __global float* output)
{
    const uint global_id = get_global_id(0);
    const uint element_offset = get_global_id(1) * INPUT_BATCH_NUM * INPUT_FEATURE_NUM;

    const uint linear_id = global_id + element_offset;
    float acc = 0;

    int input_offset_f = global_id + HELP_INPUT_OFFSET * INPUT_BATCH_NUM;
    int input_idx = input_offset_f + element_offset;
    for (int i = 0; i < P_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT_FEATURE_NUM * INPUT_BATCH_NUM;

        float value = zero ? 0 : input[input_idx];
        acc = mad(value, value, acc);

        input_offset_f+= INPUT_BATCH_NUM;
        input_idx += INPUT_BATCH_NUM;
    }
    acc = mad(acc, ALPHA, K);
    acc = native_powr(acc, -BETA);

    output[linear_id] = acc * input[linear_id];
}