#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if RELU && FP16_UNIT_USED
    #define ACTIVATION(output, input) output = max(input, 0.0h) + convert_half(NEGATIVE_SLOPE) * min(input, 0.0h);
#elif RELU
    #define ACTIVATION(output, input) output = max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f);
#else
    #define ACTIVATION(output, input) output = input;
#endif

// Required JIT constants:
//  - FP16_SUPPORTED       - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED       - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE            - Type of unit of input/output/weight/bias.
//  - UNIT_VAL_ZERO        - Literal of current UNIT_TYPE that represents 0.
//  - INPUT_BATCH_NUM      - [int] Number of elements from single spatial and single feature that are grouped in single batch in input.
//  - INPUT_ELEMENTS_COUNT - [int] Cumulative number of elements from input that are processed in single batch.
//  - WEIGHTS_BATCH_NUM    - [int] Cumulative number of elements that are outputted in single batch.
//  - RELU                 - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE       - [float] Factor for negative output values (required when ReLU is specified).


KERNEL (fully_connected_gpu_yxfn)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weight,
    const __global UNIT_TYPE* bias)
{
    const uint x = get_global_id(0);
    const uint batch_id = x % INPUT_BATCH_NUM;
    const uint neuronIdx = x / INPUT_BATCH_NUM;

    UNIT_TYPE result = bias[neuronIdx];

    uint weight_offset = neuronIdx * INPUT_FEATURE_NUM * INPUT_SIZE_Y * INPUT_SIZE_X;
    for (uint k = 0; k < INPUT_FEATURE_NUM; k++)
    {
        for (uint j = 0; j < INPUT_SIZE_Y; j++)
        {
            for(uint i = 0; i < INPUT_SIZE_X; i++)
            {
                result += input[(k + INPUT_FEATURE_NUM * (i + j * INPUT_SIZE_X)) * INPUT_BATCH_NUM + batch_id] * weight[weight_offset++];
            }
        }
    }
    ACTIVATION(output[x], result);
}

#undef ACTIVATION