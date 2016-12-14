KERNEL (reorder_gpu_padding_bfyx_f32)(const __global float* input, __global float* output)
{
    const uint pos_b = get_global_id(0);
    const uint pos_f = get_global_id(1);
    const uint pos_y = get_global_id(2);

    uint input_pos = pos_b * INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_FEATURE_NUM;
    input_pos += pos_f * INPUT_SIZE_X * INPUT_SIZE_Y;
    input_pos += pos_y * INPUT_SIZE_X;

    uint output_pos = (pos_b * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_FEATURE_NUM);
    output_pos += pos_f * OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
    output_pos += (pos_y + PADDING[3]) * OUTPUT_SIZE_X;
    output_pos += PADDING[2];
    for(uint x = 0; x < INPUT_SIZE_X; x++)
    {
        output[output_pos++] = input[input_pos++];
    }
}

#undef SRC_DEST_TYPE_CVT_FUNC
#undef TYPE_CVT_FUNC2
#undef TYPE_CVT_FUNC3