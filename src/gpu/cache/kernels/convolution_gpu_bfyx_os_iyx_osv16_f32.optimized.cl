#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

#define SIMD_SIZE 16

// FEATURES_THREADS_PER_BATCH defines how many threads are processing single batch.
// ideally, each thread should process one output feature, however since threads are
// stack in groups of 16 (SIMD16) when number of output feature maps is not dividable by 16
// there are dummy threads added in count of LEFTOVERS. We need to take them into consideration while
// calculating batch id (see lines 33-34).
#ifdef LEFTOVERS
#define FEATURES_THREADS_PER_BATCH (FILTER_OUTPUT_FEATURE_NUM + LEFTOVERS)
#else
#define FEATURES_THREADS_PER_BATCH (FILTER_OUTPUT_FEATURE_NUM)
#endif

__attribute__((intel_reqd_sub_group_size(SIMD_SIZE))) // Why driver gives us warning here?
KERNEL(convolution_gpu_bfyx_os_iyx_osv16_f32)(
    const __global float* input,
    __global float* output,
    const __global float* weights,
    const __global float* bias,
    uint split_idx)
{
    uint oc  = get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column 
    uint or  = get_global_id(1) * OUT_BLOCK_HEIGHT; // or = Output Row
    uint fm  = get_global_id(2);                    // fm = Feature Map = od = Output Depth 
    uint lid = get_local_id(2); 
    
    uint batch_idx = fm / FEATURES_THREADS_PER_BATCH;
    uint feature_idx = fm % FEATURES_THREADS_PER_BATCH;
    uint fmg = feature_idx / SIMD_SIZE;

    float in[INPUT_BLOCK_LEN];
    float out[OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT];
    float w[PREFETCH];
    uint in_addr;
    uint weight_addr = fmg * FILTER_INPUT_FEATURE_NUM * FILTER_SIZE_X * FILTER_SIZE_Y * SIMD_SIZE + lid;
    
    for(int i = 0; i < (OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT); i++) { 
        out[i] = 0.0f;
    }

    uint in_split_offset = split_idx * (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X) * FILTER_INPUT_FEATURE_NUM;
    in_addr = batch_idx * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X) * (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * INPUT_FEATURE_NUM;
    in_addr += in_split_offset + (INPUT_PADDING_SIZE_Y + INPUT_OFFSET_SIZE_Y + or * STRIDE_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X) + (INPUT_PADDING_SIZE_X + INPUT_OFFSET_SIZE_X + oc * STRIDE_SIZE_X) + lid;

    for(int kd = 0; kd < FILTER_INPUT_FEATURE_NUM; kd++)  // _ID = 3, RGB
    {
        uint tmp_in_addr = in_addr;

#if STRIDE_SIZE_X > 2   //for stride 4, INPUT_BLOCK_LEN should have value 30, we are reading 3 inputs per loop
        // read 24x19 block into registers.
        // This is ugly, we really need to fix the programming model.
        for(uint reg = 0; reg < INPUT_BLOCK_LEN; reg+=3) {
            in[reg] = input[tmp_in_addr];// read 16 elements
            // might be better to adjust the addrs, then do single load.
            if(lid < 8) in[reg + 1] = input[tmp_in_addr + 16];// read 8 elements in lower portion, for total of 24 from input row.
            tmp_in_addr += (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X);  // move to next row down 
            if(lid >= 8) in[reg + 1] = input[tmp_in_addr - 8];  // read 8 elements into upper portion
            in[reg + 2] = input[tmp_in_addr + 8]; // read 16 elements
            tmp_in_addr += (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X);  // move to next row down 
        }
#else
        for(uint reg = 0; reg < INPUT_BLOCK_LEN; ++reg) {
            in[reg] = input[tmp_in_addr];// read 16 elements
            tmp_in_addr += (INPUT_SIZE_X +  2 * INPUT_PADDING_SIZE_X);  // move to next row down 
        }
#endif

        //move to next filter
        in_addr += (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X);

        for(int pf=0; pf<PREFETCH; pf++) {
            w[pf] = weights[weight_addr]; weight_addr += SIMD_SIZE;
        }

        int wi=0;
        int kr = 0; // kr = Kernel Row
        LOOP(FILTER_SIZE_Y, kr,  // LOOP is a macro that unrolls the loop.
        {
            int kc = 0; // kc = Kernel Column
            LOOP(FILTER_SIZE_X, kc,
            {
                //w = weights[weight_addr];	
                for(uint br=0; br<OUT_BLOCK_HEIGHT; br++) {
                    for(uint bc=0; bc<OUT_BLOCK_WIDTH; bc++) {

#if STRIDE_SIZE_X > 2   //for stride 4 there's slighty different shuffling method
                        //if we fix the programming model, then we could use a nice simple 2d array: val = in[br * STRIDE_SIZE_Y + kr][bc * STRIDE_SIZE_X + kc]; 
                        float val = intel_sub_group_shuffle( in[(((br*STRIDE_SIZE_Y+kr)*24)+(bc * STRIDE_SIZE_X + kc)) / SIMD_SIZE], (((br*STRIDE_SIZE_Y+kr)*24)+(bc * STRIDE_SIZE_X + kc)) & (SIMD_SIZE - 1));
#else
                        float val = intel_sub_group_shuffle( in[br * STRIDE_SIZE_X + kr], bc * STRIDE_SIZE_X + kc);	
#endif

                        out[br * OUT_BLOCK_WIDTH + bc] = mad(w[wi % PREFETCH], val, out[br * OUT_BLOCK_WIDTH + bc]);
                    }
                }
                w[wi % PREFETCH] = weights[weight_addr];
                weight_addr += SIMD_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
                wi++;
            });
        });
        // addr went beyond due to prefetch so move it back to correct location.
        weight_addr -= PREFETCH * SIMD_SIZE;
    }

    uint out_split_offset = split_idx * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) * FILTER_OUTPUT_FEATURE_NUM;
    uint out_addr = batch_idx * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y) * OUTPUT_FEATURE_NUM;
    out_addr += out_split_offset + feature_idx * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y); // out_addr indices into start of 16 feature maps.
    out_addr += (OUTPUT_PADDING_SIZE_Y + or) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) + OUTPUT_PADDING_SIZE_X + oc;  // offset for the 4x3 block that this workitem is working on;

    for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
            out[r * OUT_BLOCK_WIDTH + c] += bias[feature_idx];
        }
    }

    for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
            ACTIVATION(out[r * OUT_BLOCK_WIDTH + c], out[r * OUT_BLOCK_WIDTH + c]);
        }
    }

#ifdef LEFTOVERS
    if (feature_idx < OUTPUT_FEATURE_NUM)
#endif
    for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
        if(!(or + r >= OUTPUT_SIZE_Y))
        {
            for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
                // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
                if(!(oc + c >= OUTPUT_SIZE_X))
                    output[out_addr + r * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) + c] = out[r * OUT_BLOCK_WIDTH + c];
            }
        }
    }
}

#undef FEATURES_THREADS_PER_BATCH

#undef ACTIVATION