#ifdef RELU
#define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
#define ACTIVATION(output, input) output = input;
#endif

// each work-item iterates this many times in the width dimension
#define OUT_BLOCK_WIDTH 6  
// each work-itme iterates this many times in the height dimension
#define OUT_BLOCK_HEIGHT 4

#define SIMD_SIZE 16

#define RADIUS (FILTER_SIZE_X-1)/2
#define PREFETCH 4

__attribute__((intel_reqd_sub_group_size(SIMD_SIZE))) // Why driver gives us warning here?
KERNEL(convolution_gpu_bfyx_os_iyx_osv16_b8_f32_stride1)(
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
    
    uint batch_idx = fm / FILTER_OUTPUT_FEATURE_NUM;
    uint feature_idx = fm % FILTER_OUTPUT_FEATURE_NUM;
    uint fmg = feature_idx / SIMD_SIZE;

    float in[8]; // need 8x10 block of input data for 4x6 outputs with 5x5 kernel stride 1.
    float out[OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT]; // 4x9 block of outputs that is SIMD_SIZE deep (along the Feature Map dimension).
    float w[PREFETCH];
    
    uint weight_addr = fmg * FILTER_INPUT_FEATURE_NUM * FILTER_SIZE_X * FILTER_SIZE_Y * SIMD_SIZE + lid;

    for(int i=0;i<(OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT);i++) {  // todo: init these with the neural net biases. 	
        out[i]=0.0f;
    }

    uint in_split_offset = split_idx * (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X) * FILTER_INPUT_FEATURE_NUM;
    uint in_addr = batch_idx * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X) * (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * INPUT_FEATURE_NUM;
    in_addr += in_split_offset + (INPUT_PADDING_SIZE_Y + INPUT_OFFSET_SIZE_Y + or * STRIDE_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X) + (INPUT_PADDING_SIZE_X + INPUT_OFFSET_SIZE_X + oc * STRIDE_SIZE_X) + lid;
    
    for(int kd = 0; kd < FILTER_INPUT_FEATURE_NUM; kd++)
    {
        uint tmp_in_addr = in_addr;

        for(uint reg = 0; reg < ((OUT_BLOCK_HEIGHT * STRIDE_SIZE_X) + RADIUS + RADIUS); reg++) {
            in[reg] = input[tmp_in_addr];// read 16 elements
            tmp_in_addr += (INPUT_SIZE_X +  2 * INPUT_PADDING_SIZE_X);  // move to next row down 
        }
        
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
                for(int br=0; br<OUT_BLOCK_HEIGHT; br++) {
                    for(int bc=0; bc<OUT_BLOCK_WIDTH; bc++) {
                        float val = intel_sub_group_shuffle( in[br * STRIDE_SIZE_X + kr], bc * STRIDE_SIZE_X + kc);	
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

    // write the 4x3 (and 16 feature maps deep) output tile to memory
    uint out_split_offset = split_idx * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) * FILTER_OUTPUT_FEATURE_NUM;
    uint out_addr = batch_idx * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y) * OUTPUT_FEATURE_NUM;
    out_addr += out_split_offset + feature_idx * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) * (OUTPUT_SIZE_Y + 2 * OUTPUT_PADDING_SIZE_Y); // out_addr indexes into start of 16 feature maps.
    out_addr += (OUTPUT_PADDING_SIZE_Y + or) * (OUTPUT_SIZE_X + 2 * OUTPUT_PADDING_SIZE_X) + OUTPUT_PADDING_SIZE_X + oc;  // offset for the 4x3 block that this workitem is working on;

    for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
            out[r * OUT_BLOCK_WIDTH + c] += bias[fmg * 16 + lid];
        }
    }

    for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
            ACTIVATION(out[r * OUT_BLOCK_WIDTH + c], out[r * OUT_BLOCK_WIDTH + c]);
        }
    }

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

#undef PREFETCH
#undef OUT_BLOCK_WIDTH
#undef OUT_BLOCK_HEIGHT

#undef ACTIVATION