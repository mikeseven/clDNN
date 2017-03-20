#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if FP16_UNIT_USED
    #define UNIT_CVT_FUNC(val) convert_half(val)
#else
    #define UNIT_CVT_FUNC(val) (val)
#endif
 

KERNEL (lrn_gpu_within_channel_yxfb)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
	const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
	const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;

    for (uint index = get_global_id(0) ; index < COUNT ; index += get_global_size(0)) 
    {        
        const uint b  = index % OUTPUT_BATCH_NUM;    
        const uint fm  = (index / OUTPUT_BATCH_NUM) % OUTPUT_FEATURE_NUM;
        const uint x = (index / OUTPUT_BATCH_NUM / OUTPUT_FEATURE_NUM) % OUTPUT_SIZE_X;  
        const uint y = index / OUTPUT_BATCH_NUM / OUTPUT_FEATURE_NUM / OUTPUT_SIZE_X;
        int hstart = y - PAD;
        int wstart = x - PAD;
        int hend = min(hstart + P_SIZE, INPUT_SIZE_Y + PAD);
        int wend = min(wstart + P_SIZE, INPUT_SIZE_X + PAD);
        const int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, (int)0);
        wstart = max(wstart, (int)0);
        hend = min(hend, INPUT_SIZE_Y);
        wend = min(wend, INPUT_SIZE_X);
        UNIT_TYPE aveval = 0;
  
        for (int h = hstart; h < hend; h++) 
        {
            for (int w = wstart; w < wend; w++) 
            {
                int offset = h * OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM * OUTPUT_SIZE_X + w * OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM + fm * OUTPUT_BATCH_NUM + b;
                UNIT_TYPE tmp_val = input[offset] * UNIT_CVT_FUNC(ALPHA_VAL_FACTOR);
                aveval += (tmp_val * tmp_val);
            }
        }
            
        UNIT_TYPE acc = aveval / pool_size;
        acc = mad(acc, UNIT_CVT_FUNC(ALPHA), UNIT_CVT_FUNC(K));
        acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));
		
		uint output_pos = b + OUTPUT_BATCH_NUM * (fm + OUTPUT_FEATURE_NUM * ((OUTPUT_PADDING_LOWER_SIZE_Y + y) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X + x));

        output[output_pos] = acc * input[index];
    }    
}