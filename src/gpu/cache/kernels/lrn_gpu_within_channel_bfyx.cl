#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if FP16_UNIT_USED
    #define UNIT_CVT_FUNC(val) convert_half(val)
#else
    #define UNIT_CVT_FUNC(val) (val)
#endif


KERNEL (lrn_gpu_within_channel_bfyx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;

    for (uint index = get_global_id(0) ; index < COUNT ; index += get_global_size(0)) 
    {
        const uint pw = index % OUTPUT_SIZE_X;
        const uint ph = (index / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
        const uint c = (index / OUTPUT_SIZE_X / OUTPUT_SIZE_Y) % OUTPUT_FEATURE_NUM;
        const uint n = index / OUTPUT_SIZE_X / OUTPUT_SIZE_Y / OUTPUT_FEATURE_NUM;
        int hstart = ph - PAD;
        int wstart = pw - PAD;
        int hend = min(hstart + P_SIZE, INPUT_SIZE_Y + PAD);
        int wend = min(wstart + P_SIZE, INPUT_SIZE_X + PAD);
        const int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, (int)0);
        wstart = max(wstart, (int)0);
        hend = min(hend, INPUT_SIZE_Y);
        wend = min(wend, INPUT_SIZE_X);
        UNIT_TYPE aveval = 0;
        __global const UNIT_TYPE* bottom_slice = input + (n * OUTPUT_FEATURE_NUM + c) * INPUT_SIZE_Y * INPUT_SIZE_X;
        for (int h = hstart; h < hend; ++h) 
        {
            for (int w = wstart; w < wend; ++w) 
            {
                UNIT_TYPE tmp_val = bottom_slice[h * INPUT_SIZE_X + w] * UNIT_CVT_FUNC(ALPHA_VAL_FACTOR);
                aveval += (tmp_val * tmp_val);
            }
        }
            
        UNIT_TYPE acc = aveval / pool_size;
        acc = mad(acc, UNIT_CVT_FUNC(ALPHA), UNIT_CVT_FUNC(K));
        acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));

        uint output_pos = (n * OUTPUT_FEATURE_NUM + c) * output_buffer_size_x * output_buffer_size_y;
        output_pos += (OUTPUT_PADDING_LOWER_SIZE_Y + ph) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X + pw;

        output[output_pos] = acc * input[index];
    }    
}