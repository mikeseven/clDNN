#if RELU
    #define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
    #define ACTIVATION(output, input) output = input;
#endif

KERNEL(convolution_gpu_yxfb_yxio)(
    const __global float* input,
    __global float* output,
    const __global float* filter,
    const __global float* bias,
    uint split_idx)
{
    const int batch_num = INPUT_BATCH_NUM;

    const uint linear_id = get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2));
    const int bifn_num = batch_num * FILTER_OUTPUT_FEATURE_NUM;
    int global_id = linear_id % bifn_num + (linear_id / bifn_num) * bifn_num * FILTER_ARRAY_NUM + split_idx * bifn_num;

    const int ofm_offset = (global_id / batch_num) % FILTER_OUTPUT_FEATURE_NUM;

    float result = bias[ofm_offset];

    bool finish = false;
    const uint out_x = get_global_id(1);
    const uint out_y = get_global_id(2);

    finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_OFFSET_SIZE_X;
    finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_OFFSET_SIZE_Y) ? true : finish;
	
    if(!finish)
    {
        const int batch_offset = global_id % batch_num;

        const int x = out_x - INPUT_OFFSET_SIZE_X - (FILTER_SIZE_X - 1);
        const int y = out_y - INPUT_OFFSET_SIZE_Y - (FILTER_SIZE_Y - 1);

        for (uint i = 0; i < FILTER_SIZE_Y; i++)
        {
            int input_offset_y = y + i;
            bool zero_y = (input_offset_y >= INPUT_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0) || ((input_offset_y % STRIDE_SIZE_Y) != 0);
            if(!zero_y)
            {
                for (uint j = 0; j < FILTER_SIZE_X; j++)
                {
                    int input_offset_x = x + j;
                    bool zero = (input_offset_x >= INPUT_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0) || ((input_offset_x % STRIDE_SIZE_X) != 0);
                    if(!zero)
                    {
                        int input_idx = (input_offset_x / STRIDE_SIZE_X + (input_offset_y * INPUT_SIZE_X / STRIDE_SIZE_Y)) * INPUT_FEATURE_NUM * batch_num;
                        input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * batch_num;
                        input_idx += batch_offset;

                        uint filter_idx = ofm_offset + FILTER_INPUT_FEATURE_NUM * FILTER_OUTPUT_FEATURE_NUM * ((FILTER_SIZE_X * FILTER_SIZE_Y - 1) - (i * FILTER_SIZE_X + j));


                        for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
                        {
                            result = mad(input[input_idx], filter[filter_idx], result);
                            filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            input_idx += batch_num;
                        }
                    }
                } 
            }
        }
    }
	ACTIVATION(output[global_id], result);
}

#undef ACTIVATION