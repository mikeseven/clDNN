// Extensions and additional capabilities.
#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// ---------------------------------------------------------------------------------------------------------------------

// Activation function used in ReLU.
#if RELU && FP16_UNIT_USED
    #define ACTIVATION(output, input) output = max(input, 0.0h) + convert_half(NEGATIVE_SLOPE) * min(input, 0.0h);
#elif RELU
    #define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
    #define ACTIVATION(output, input) output = input;
#endif

// ---------------------------------------------------------------------------------------------------------------------

// Macro loop implementation (max. count = 32).
#define LOOP(statement, count) LOOP_HANDLER(statement, count)
#define LOOP_HANDLER(statement, count) LOOP_##count(statement, 0, _i)

#define LOOP_32(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_31(statement, (index + 1), suffix##i)

#define LOOP_31(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_30(statement, (index + 1), suffix##i)

#define LOOP_30(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_29(statement, (index + 1), suffix##i)

#define LOOP_29(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_28(statement, (index + 1), suffix##i)

#define LOOP_28(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_27(statement, (index + 1), suffix##i)

#define LOOP_27(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_26(statement, (index + 1), suffix##i)

#define LOOP_26(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_25(statement, (index + 1), suffix##i)

#define LOOP_25(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_24(statement, (index + 1), suffix##i)

#define LOOP_24(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_23(statement, (index + 1), suffix##i)

#define LOOP_23(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_22(statement, (index + 1), suffix##i)

#define LOOP_22(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_21(statement, (index + 1), suffix##i)

#define LOOP_21(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_20(statement, (index + 1), suffix##i)

#define LOOP_20(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_19(statement, (index + 1), suffix##i)

#define LOOP_19(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_18(statement, (index + 1), suffix##i)

#define LOOP_18(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_17(statement, (index + 1), suffix##i)

#define LOOP_17(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_16(statement, (index + 1), suffix##i)

#define LOOP_16(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_15(statement, (index + 1), suffix##i)

#define LOOP_15(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_14(statement, (index + 1), suffix##i)

#define LOOP_14(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_13(statement, (index + 1), suffix##i)

#define LOOP_13(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_12(statement, (index + 1), suffix##i)

#define LOOP_12(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_11(statement, (index + 1), suffix##i)

#define LOOP_11(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_10(statement, (index + 1), suffix##i)

#define LOOP_10(statement, index, suffix)      \
    statement(index, suffix)                   \
    LOOP_9(statement, (index + 1), suffix##i)

#define LOOP_9(statement, index, suffix)       \
    statement(index, suffix)                   \
    LOOP_8(statement, (index + 1), suffix##i)

#define LOOP_8(statement, index, suffix)       \
    statement(index, suffix)                   \
    LOOP_7(statement, (index + 1), suffix##i)

#define LOOP_7(statement, index, suffix)       \
    statement(index, suffix)                   \
    LOOP_6(statement, (index + 1), suffix##i)

#define LOOP_6(statement, index, suffix)       \
    statement(index, suffix)                   \
    LOOP_5(statement, (index + 1), suffix##i)

#define LOOP_5(statement, index, suffix)       \
    statement(index, suffix)                   \
    LOOP_4(statement, (index + 1), suffix##i)

#define LOOP_4(statement, index, suffix)       \
    statement(index, suffix)                   \
    LOOP_3(statement, (index + 1), suffix##i)

#define LOOP_3(statement, index, suffix)       \
    statement(index, suffix)                   \
    LOOP_2(statement, (index + 1), suffix##i)

#define LOOP_2(statement, index, suffix)       \
    statement(index, suffix)                   \
    LOOP_1(statement, (index + 1), suffix##i)

#define LOOP_1(statement, index, suffix)       \
    statement(index, suffix)                   \
    LOOP_0(statement, (index + 1), suffix##i)

#define LOOP_0(statement, index, suffix)

// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

// Required JIT constants:
//  - FP16_SUPPORTED       - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED       - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE            - Type of unit of input/output/weight/bias.
//  - UNIT_BYTE_SIZE       - 
//  - UNIT_VAL_ZERO        - Literal of current UNIT_TYPE that represents 0.
//  - INPUT_BATCH_NUM      - [int] Batch size for input. Number of input sets of spatial and feature data that are grouped
//                           to be processed in single batch.
//  - INPUT_ELEMENTS_COUNT - [int] Cumulative number of elements in single data set from batch.
//  - WEIGHTS_BATCH_NUM    - [int] Cumulative number of elements that are outputted for single input set from batch.
//  - RELU                 - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE       - [float] Factor for negative output values (required when ReLU is specified).
//
//  - SUB_GROUP_SIZE       - [int] Size of used subgroup (SIMD).
//  - WORK_ITEMS_PER_BATCH - [int] Number of work items needed to process at least one element in all data sets from batch.
//  - UNIT_BYTE_SIZE       - [int] Size of unit of input/output/weight/bias in bytes.
//  - CHUNK_TYPE           - Type of chunk of data read by work item using sub-group operation.
//  - CHUNK_BYTE_SIZE      - [int] Size of chunk of data read by work item using sub-group operation in bytes.
//  - UNITS_PER_CHUNK      - [int] Number of units stored in single chunk of read data.
//  - BYTES_PER_SG_READ    - [int] Number of bytes read by single sub-group read operation (read by entire sub-group).
//  - UNITS_PER_SG_READ    - [int] Number of units read by single sub-group read operation (read by entire sub-group).


// Currently block read is 4 bytes aligned.
#define ALIGNED_BLOCK_READ(ptr, byte_offset) intel_sub_group_block_read((const __global uint*)((const __global char*)(ptr) + (byte_offset)))
// Currently block write is 16 bytes aligned.
#define ALIGNED_BLOCK_WRITE(ptr, byte_offset, val) intel_sub_group_block_write((__global uint*)((__global char*)(ptr) + (byte_offset)), (val))

// Depends on batch size.
#define INPUT_READ(ptr, byte_offset) ALIGNED_BLOCK_READ(ptr, byte_offset)
// Depends on number of responses.
#define FILTER_READ(ptr, byte_offset) ALIGNED_BLOCK_READ(ptr, byte_offset)
// Depends on BYTES_PER_SG_READ.
#define BIAS_READ(ptr, byte_offset) ALIGNED_BLOCK_READ(ptr, byte_offset)
// Depends on batch size.
#define OUTPUT_WRITE(ptr, byte_offset, val) ALIGNED_BLOCK_WRITE(ptr, byte_offset, val)

/*
#if WEIGHTS_BATCH_NUM % (2 * SUB_GROUP_SIZE) == 0 || (!FP16_UNIT_USED && WEIGHTS_BATCH_NUM % SUB_GROUP_SIZE == 0)
    #define FILTER_READ(ptr, byte_offset) ALIGNED_BLOCK_READ(ptr, byte_offset)
#elifs
    #define FILTER_READ(ptr, byte_offset) ALIGNED_BLOCK_READ(ptr, byte_offset)
#elif WEIGHTS_BATCH_NUM % 8 == 0
#else
#endif




#if FP16_UNIT_USED
    #define ALIGNED_FILTER_BLOCK_READ(ptr, byte_offset) as_half2(intel_sub_group_block_read((const __global uint*)((const __global char*)(ptr) + (byte_offset))))
    #define FILTER_TYPE half2
#else
    #define ALIGNED_FILTER_BLOCK_READ(ptr, byte_offset) as_float(intel_sub_group_block_read((const __global uint*)((const __global char*)(ptr) + (byte_offset))))
    #define FILTER_TYPE float
#endif
*/


#if INPUT_BATCH_NUM > 0 && INPUT_BATCH_NUM % (SUB_GROUP_SIZE * CHUNK_BYTE_SIZE / UNIT_BYTE_SIZE) == 0
#else
    #error Kernel does not support specified input batch size.
#endif



__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL (fully_connected_gpu_xb_xb_block_fp16)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weight,
    const __global UNIT_TYPE* bias)
{
    // constexpr:
    const uint input_batch_byte_size       = INPUT_BATCH_NUM * UNIT_BYTE_SIZE;
    const uint input_byte_size             = INPUT_ELEMENTS_COUNT * input_batch_byte_size;
    const uint input_yxf_elems_per_sg_read = INPUT_BATCH_NUM < UNITS_PER_SG_READ
                                               ? UNITS_PER_SG_READ / INPUT_BATCH_NUM
                                               : 1;
    const uint input_sg_reads_distance     = WORK_ITEMS_PER_BATCH * BYTES_PER_SG_READ;

    // Size in bytes of all responses for single spatial/feature data point (the same as filter_yxf_elems_distance).
    // Distance between two nearest xyf elements with the same response id.
    const uint filter_response_byte_size = WEIGHTS_BATCH_NUM * UNIT_BYTE_SIZE;
    // Cumulative size in bytes of all weights/filters.
    const uint filters_byte_size         = INPUT_ELEMENTS_COUNT * filter_response_byte_size;

    const uint output_batch_byte_size = input_batch_byte_size;
    const uint output_byte_size = WEIGHTS_BATCH_NUM * output_batch_byte_size;

// ---------------------------------------------------------------------------------------------------------------------

    // non-constexpr:
    // Identifier of processing sub-group (each sub-group process UNITS_PER_SG_READ output responses for at least
    // one data set in batch).
    const uint sg_id          = get_group_id(0);
    // Identifier of batch group (each batch group process up to UNITS_PER_SG_READ data sets from batch).
    const uint batch_group_id = get_global_id(1);
    // Identifier of work item element in processing sub-group.
    const uint sg_elem_id     = get_sub_group_local_id();

    // Input base offset in bytes (yxfb/xb format of input).
    const uint input_base     = batch_group_id * BYTES_PER_SG_READ;

    // Filter base offset in bytes (yxfb/xb format of weights).
    const uint filter_base    = sg_id * BYTES_PER_SG_READ;

    // Output base offset in bytes (xb format of output).
    const uint output_base    = (sg_id * output_batch_byte_size + batch_group_id) * BYTES_PER_SG_READ;
    const uint output_limit   = min(output_base + UNITS_PER_SG_READ * output_batch_byte_size, output_byte_size);

    // Filter/input byte offsets in sub-group used duering read/write operations.
    const uint sg_elem_offset = sg_elem_id * CHUNK_BYTE_SIZE;


    // Accumulator over batch and response elements.
    CHUNK_TYPE acc[UNITS_PER_SG_READ] = {};

    // Iterate over yxf linear plane (both filters/weights and input).
    for (uint input_offset = input_base, filter_offset = filter_base; input_offset < input_byte_size; input_offset += input_sg_reads_distance)
    {
        CHUNK_TYPE input_val = INPUT_READ(input, input_offset + sg_elem_offset);

        // Iterate over filters needed to process input read by sub-group.
        for(uint elem_idx = 0; elem_idx < input_yxf_elems_per_sg_read; ++elem_idx)
        {
            CHUNK_TYPE filter_val = FILTER_READ(weight, filter_offset + sg_elem_offset);
            filter_offset += filter_response_byte_size;

            // MULTIPLY

            // BATCH = 32x? (HF) / 16x? (F)
            #pragma unroll
            for (uint acc_pos = 0; acc_pos < UNITS_PER_SG_READ; acc_pos += UNITS_PER_CHUNK)
            {
#if FP16_UNIT_USED
                acc[acc_pos]     = as_uint(fma(as_half2(input_val), as_half2(intel_sub_group_shuffle(filter_val, acc_pos)).s0, as_half2(acc[acc_pos])));
                acc[acc_pos + 1] = as_uint(fma(as_half2(input_val), as_half2(intel_sub_group_shuffle(filter_val, acc_pos)).s1, as_half2(acc[acc_pos + 1])));
#else
                acc[acc_pos] = as_uint(fma(as_float(input_val), as_float(intel_sub_group_shuffle(filter_val, acc_pos)), as_float(acc[acc_pos])));
#endif
            }
        }
    }

    // WRITE OUTPUT
    // BATCH = 32x? (HF) / 16x? (F)
    {
        uint output_offset = output_base;
        for (uint acc_pos = 0; acc_pos < UNITS_PER_SG_READ; ++acc_pos)
        {
#if FP16_UNIT_USED
            half2 output_val = as_half2(acc[acc_pos]) + bias[sg_id * UNITS_PER_SG_READ + acc_pos];
#else
            float output_val = as_float(acc[acc_pos]) + bias[sg_id * UNITS_PER_SG_READ + acc_pos];
#endif
            ACTIVATION(output_val, output_val)
            if (output_offset < output_limit)
            {
                OUTPUT_WRITE(output, output_offset + sg_elem_offset, as_uint(output_val));
            }
            output_offset += output_batch_byte_size;
        }
    }
}

#undef ACTIVATION

#undef LOOP
#undef LOOP_HANDLER
#undef LOOP_32
#undef LOOP_31
#undef LOOP_30
#undef LOOP_29
#undef LOOP_28
#undef LOOP_27
#undef LOOP_26
#undef LOOP_25
#undef LOOP_24
#undef LOOP_23
#undef LOOP_22
#undef LOOP_21
#undef LOOP_20
#undef LOOP_19
#undef LOOP_18
#undef LOOP_17
#undef LOOP_16
#undef LOOP_15
#undef LOOP_14
#undef LOOP_13
#undef LOOP_12
#undef LOOP_11
#undef LOOP_10
#undef LOOP_9
#undef LOOP_8
#undef LOOP_7
#undef LOOP_6
#undef LOOP_5
#undef LOOP_4
#undef LOOP_3
#undef LOOP_2
#undef LOOP_1
#undef LOOP_0

#undef ALIGNED_BLOCK_READ
#undef ALIGNED_BLOCK_WRITE
#undef INPUT_READ
#undef FILTER_READ
#undef BIAS_READ
#undef OUTPUT_WRITE
