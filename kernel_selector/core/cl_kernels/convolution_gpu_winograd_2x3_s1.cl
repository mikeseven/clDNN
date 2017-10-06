/*/// -----------------------------------------------------------------------------------------------------------------------------
Copyright (c) 2016, Intel Corporation
/*/// -----------------------------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------------------------------------
// L3_SIMD_4x8
// Input matrices dimensions: M x K x N
// Output matrix dimensions: M x N
// --------------------------------------------------------------------------------------------------------------------------------

__attribute__((reqd_work_group_size(8, 1, 1)))
void kernel KERNEL(convolution_gpu_winograd_2x3_s1)
(
    const __global UNIT_TYPE *signalw,
    const __global UNIT_TYPE *filterw,
          __global UNIT_TYPE *outputw
#ifdef BIAS_TERM
    , const __global UNIT_TYPE* bias
#endif
)
{

};
