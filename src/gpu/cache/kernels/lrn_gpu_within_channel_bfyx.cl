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
    for (int index = get_global_id(0) ; index < COUNT ; index += get_global_size(0)) 
    {
        const int pw = index % OUTPUT_SIZE_X;
        const int ph = (index / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
        const int c = (index / OUTPUT_SIZE_X / OUTPUT_SIZE_Y) % OUTPUT_FEATURE_NUM;
        const int n = index / OUTPUT_SIZE_X / OUTPUT_SIZE_Y / OUTPUT_FEATURE_NUM;
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
                UNIT_TYPE tmp_val = bottom_slice[h * INPUT_SIZE_X + w];
                aveval += (tmp_val * tmp_val);
            }
        }
            
        UNIT_TYPE acc = aveval / pool_size;
        acc = mad(acc, UNIT_CVT_FUNC(ALPHA), UNIT_CVT_FUNC(K));
        acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));

        output[index] = acc * input[index];
    }    
}