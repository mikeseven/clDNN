// Extensions and additional capabilities.
#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Just-in-time macro definitions:
// ---------------------------------------------------------------------------------------------------------------------

// Required JIT constants:
//  - INPUT                - [tensor] Input dimensions (batch, spatial and feature).
//  - OUTPUT               - [tensor] Output dimensions (batch, spatial and feature).
//  - STRIDE               - [tensor] Stride (only spatial). Factors that describe step size in X or Y dimension of
//                           input position of application of convolution filter when next ouput value
//                           (step 1 in in X or Y dimension of output) is computed.
//  - INPUT_OFFSET         - [tensor] Offset between input and output (only spatial). Non-positive values that describe
//                           initial offset input position of application of convolution filter and output position.
//  - OUTPUT_OFFSET        - [tensor] Offset of ouput writes (only spatial). Non-negative values that describe
//                           shift (in X and Y) of output writes for output position.
//  - FP16_SUPPORTED       - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED       - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE            - Type of unit of input/output/weight/bias.
//  - UNIT_VAL_ZERO        - Literal of current UNIT_TYPE that represents 0.
//  - RELU                 - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE       - [float] Factor for negative output values (required when ReLU is specified).
//
//  - SUB_GROUP_SIZE       - [int] Size of used subgroup (SIMD).
//  - LEFTOVERS            - [int] Optional parameter, required only when number of ofm is not dividable by SUB_GROUP_SIZE
//                           see comment for FEATURES_THREADS_PER_BATCH for more informations

/*
gpu::make_jit_constant("OUTPUT_LIMIT",              output_size),
gpu::make_jit_constant("INPUT_PADDING",             input_padding.size()),
gpu::make_jit_constant("OUTPUT_PADDING",            outer.argument.output_padding().size()),
gpu::make_jit_constant("FILTER",                    filter_mem.argument().size),
gpu::make_jit_constant("FILTER_ARRAY_NUM",          split),
gpu::make_jit_constant("FILTER_OUTPUT_FEATURE_NUM", "FILTER_FEATURE_NUM_0"),
gpu::make_jit_constant("FILTER_INPUT_FEATURE_NUM",  "FILTER_FEATURE_NUM_1"),
gpu::make_jit_constant("OUT_BLOCK_WIDTH",           _kernel_data.block_width));
gpu::make_jit_constant("OUT_BLOCK_HEIGHT",          _kernel_data.block_height));
gpu::make_jit_constant("IN_BLOCK_ARRAY_SIZE",       _kernel_data.input_block_array_size));
gpu::make_jit_constant("IN_BLOCK_WIDTH",            _kernel_data.input_block_width));
gpu::make_jit_constant("PREFETCH",                  _kernel_data.prefetch));
if (_kernel_data.leftovers)
    gpu::make_jit_constant("LEFTOVERS",             _kernel_data.leftovers));
*/

// ---------------------------------------------------------------------------------------------------------------------
// Activation mecro function:
// ---------------------------------------------------------------------------------------------------------------------

// Activation function used in ReLU.
#if RELU && FP16_UNIT_USED
    #define ACTIVATION(output, input) output = max(input, 0.0h) + convert_half(NEGATIVE_SLOPE) * min(input, 0.0h);
#elif RELU
    #define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
    #define ACTIVATION(output, input) output = input;
#endif

// FEATURES_THREADS_PER_BATCH defines how many threads in z-dimension are processing single batch.
// ideally, z-dimension of value n should indicate processing of n-th output feature. however, since
// threads are stack in groups of SUB_GROUP_SIZE, when number of ofm is not dividable by SUB_GROUP_SIZE
// there are dummy threads added in z-dimension in count of LEFTOVERS. We need to take them into consideration
// while calculating batch's id (see lines 86-87). Values calculated by dummy threads are discarded at line 210.
#ifdef LEFTOVERS
#define FEATURES_THREADS_PER_BATCH (FILTER_OUTPUT_FEATURE_NUM + LEFTOVERS)
#else
#define FEATURES_THREADS_PER_BATCH (FILTER_OUTPUT_FEATURE_NUM)
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(convolution_gpu_bfyx_os_iyx_osv16)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weights,
    const __global UNIT_TYPE* bias,
    uint split_idx)
{
    const uint oc  = get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column 
    const uint or  = get_global_id(1) * OUT_BLOCK_HEIGHT; // or = Output Row
    const uint fm  = get_global_id(2);                    // fm = Feature Map = od = Output Depth 
    const uint lid = get_sub_group_local_id();
    
    uint batch_idx = fm / FEATURES_THREADS_PER_BATCH;
    uint feature_idx = fm % FEATURES_THREADS_PER_BATCH;
    uint fmg = feature_idx / SUB_GROUP_SIZE;

    UNIT_TYPE in[IN_BLOCK_ARRAY_SIZE];
    UNIT_TYPE out[OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT];
    UNIT_TYPE w[PREFETCH];
    uint in_addr;
    uint weight_addr = fmg * FILTER_INPUT_FEATURE_NUM * FILTER_SIZE_X * FILTER_SIZE_Y * SUB_GROUP_SIZE + lid;
    
    for(int i = 0; i < (OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT); i++) { 
        out[i] = UNIT_VAL_ZERO;
    }

    uint in_split_offset = split_idx * (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X) * FILTER_INPUT_FEATURE_NUM;
    in_addr = batch_idx * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X) * (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * INPUT_FEATURE_NUM;
    in_addr += in_split_offset + (INPUT_PADDING_SIZE_Y + INPUT_OFFSET_SIZE_Y + or * STRIDE_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X) + (INPUT_PADDING_SIZE_X + INPUT_OFFSET_SIZE_X + oc * STRIDE_SIZE_X) + lid;

    for(int kd = 0; kd < FILTER_INPUT_FEATURE_NUM; kd++)  // _ID = 3, RGB
    {
        uint tmp_in_addr = in_addr;

#if IN_BLOCK_WIDTH % SUB_GROUP_SIZE == 0
        __attribute__((opencl_unroll_hint(IN_BLOCK_ARRAY_SIZE)))
        for(uint in_block_pos = 0; in_block_pos < IN_BLOCK_ARRAY_SIZE * SUB_GROUP_SIZE; in_block_pos += SUB_GROUP_SIZE) {
            // Horizontal position in input block after read.
            const uint in_block_next_x_pos = in_block_pos % IN_BLOCK_WIDTH + SUB_GROUP_SIZE;

            in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr + in_block_pos % IN_BLOCK_WIDTH];

            // If we have row break, move to the next row.
            if (in_block_next_x_pos == IN_BLOCK_WIDTH)
                tmp_in_addr += INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X;
        }
#elif (2 * IN_BLOCK_WIDTH) % SUB_GROUP_SIZE == 0
        __attribute__((opencl_unroll_hint(IN_BLOCK_ARRAY_SIZE)))
        for(uint in_block_pos = 0; in_block_pos < IN_BLOCK_ARRAY_SIZE * SUB_GROUP_SIZE; in_block_pos += SUB_GROUP_SIZE) {
            // Horizontal position in input block after read.
            const uint in_block_next_x_pos = in_block_pos % IN_BLOCK_WIDTH + SUB_GROUP_SIZE;

            if (in_block_next_x_pos <= IN_BLOCK_WIDTH) { //
                in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr + in_block_pos % IN_BLOCK_WIDTH];

                // If we have row break, move to the next row.
                if (in_block_next_x_pos == IN_BLOCK_WIDTH)
                    tmp_in_addr += INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X;
            }
            else {
                // TODO: Generalize this step to relax IN_BLOCK_WIDTH restrictions.
                // Position in sub-group on which new row need to be read.
                const uint sg_br_pos = IN_BLOCK_WIDTH - in_block_pos % IN_BLOCK_WIDTH;

                if (lid < sg_br_pos)
                    in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr + in_block_pos % IN_BLOCK_WIDTH];
                // We have row break inside sub-group. Need to move to next line.
                tmp_in_addr += INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X;
                if (lid >= sg_br_pos)
                    in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr - sg_br_pos];

                // If we have another row break, move to the next row.
                if (in_block_next_x_pos == 2 * IN_BLOCK_WIDTH)
                    tmp_in_addr += INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X;
            }
        }
#else
    #error IN_BLOCK_WIDTH must be multiple of SUB_GROUP_SIZE or half of SUB_GROUP_SIZE. Other scenarios are not currently implemented.
#endif

        //move to next filter
        in_addr += (INPUT_SIZE_Y + 2 * INPUT_PADDING_SIZE_Y) * (INPUT_SIZE_X + 2 * INPUT_PADDING_SIZE_X);

        for(int pf=0; pf<PREFETCH; pf++) {
            w[pf] = weights[weight_addr]; weight_addr += SUB_GROUP_SIZE;
        }

        uint wi = 0;
        uint kr = 0; // kr = Kernel Row
        LOOP(FILTER_SIZE_Y, kr,  // LOOP is a macro that unrolls the loop.
        {
            uint kc = 0; // kc = Kernel Column
            LOOP(FILTER_SIZE_X, kc,
            {
                //w = weights[weight_addr];	
                for(uint br=0; br<OUT_BLOCK_HEIGHT; br++) {
                    for(uint bc=0; bc<OUT_BLOCK_WIDTH; bc++) {

#if IN_BLOCK_WIDTH != SUB_GROUP_SIZE
                        //if we fix the programming model, then we could use a nice simple 2d array: val = in[br * STRIDE_SIZE_Y + kr][bc * STRIDE_SIZE_X + kc]; 
                        UNIT_TYPE val = intel_sub_group_shuffle( in[(((br * STRIDE_SIZE_Y + kr) * IN_BLOCK_WIDTH) + (bc * STRIDE_SIZE_X + kc)) / SUB_GROUP_SIZE],
                                                                    (((br * STRIDE_SIZE_Y + kr) * IN_BLOCK_WIDTH) + (bc * STRIDE_SIZE_X + kc)) % SUB_GROUP_SIZE);
#else
                        UNIT_TYPE val = intel_sub_group_shuffle( in[br * STRIDE_SIZE_X + kr], bc * STRIDE_SIZE_X + kc);	
#endif

                        out[br * OUT_BLOCK_WIDTH + bc] = mad(w[wi % PREFETCH], val, out[br * OUT_BLOCK_WIDTH + bc]);
                    }
                }
                w[wi % PREFETCH] = weights[weight_addr];
                weight_addr += SUB_GROUP_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
                wi++;
            });
        });
        // addr went beyond due to prefetch so move it back to correct location.
        weight_addr -= PREFETCH * SUB_GROUP_SIZE;
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


#undef ACTIVATION
#undef FEATURES_THREADS_PER_BATCH
