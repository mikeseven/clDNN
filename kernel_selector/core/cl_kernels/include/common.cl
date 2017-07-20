/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if FP16_UNIT_USED
    #ifndef UNIT_TYPE
    #define UNIT_TYPE half
    #endif

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    #if !defined(ACCUMULATOR_TYPE)
        #define ACCUMULATOR_TYPE float
    #endif
    
    #define UNIT_VAL_MAX HALF_MAX
    #define UNIT_VAL_MIN -UNIT_VAL_MAX
    #define UNIT_VAL_ONE 1.0h
    #define UNIT_VAL_ZERO 0.0h
    #define TO_UNIT_TYPE(v) convert_half(v)
#else
    #ifndef UNIT_TYPE
    #define UNIT_TYPE float
    #endif

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    #if !defined(ACCUMULATOR_TYPE)
        #define ACCUMULATOR_TYPE float
    #endif
    
    #define UNIT_VAL_MAX FLT_MAX
    #define UNIT_VAL_MIN -UNIT_VAL_MAX
    #define UNIT_VAL_ONE 1.0f
    #define UNIT_VAL_ZERO 0.0f
    #define TO_UNIT_TYPE(v) (float)(v)
#endif

// TODO: use native_exp and use cast for APL
#define ACTIVATION_LOGISTIC(input)                      (UNIT_VAL_ONE/(UNIT_VAL_ONE + exp(-input)))
#define ACTIVATION_HYPERBOLIC_TAN(input)                (tanh(input))
#define ACTIVATION_RELU(input)                          (fmax(UNIT_VAL_ZERO, input))
#define ACTIVATION_RELU_NEGATIVE_SLOPE(input, slope)    isinf(TO_UNIT_TYPE(slope)) ? ((input >= UNIT_VAL_ZERO) ? \
                                                        input : -TO_UNIT_TYPE(slope)) : \
                                                        (fmax(input, UNIT_VAL_ZERO) + TO_UNIT_TYPE(slope) * fmin(input, UNIT_VAL_ZERO))
#define ACTIVATION_BRELU(input, m)                      (fmax(UNIT_VAL_ZERO, fmin(m, input)))
#define ACTIVATION_SOFTRELU(input)                      (log(UNIT_VAL_ONE + exp(input)))
#define ACTIVATION_ABS(input)                           (fabs(input))
#define ACTIVATION_LINEAR(input, m, n)                  (m*input + n)
#define ACTIVATION_SQUARE(input)                        (input*input)
#define ACTIVATION_SQRT(input)                          (sqrt(input))

#if defined ACTIVATION_FUNCTION_LOGISTIC
    #define ACTIVATION(input, m, n) ACTIVATION_LOGISTIC(input)
#elif defined ACTIVATION_FUNCTION_HYPERBOLIC_TAN
    #define ACTIVATION(input, m, n) ACTIVATION_HYPERBOLIC_TAN(input)
#elif defined ACTIVATION_FUNCTION_RELU
    #define ACTIVATION(input, m, n) ACTIVATION_RELU(input)
#elif defined ACTIVATION_FUNCTION_RELU_NEGATIVE_SLOPE
    #define ACTIVATION(input, m, n) ACTIVATION_RELU_NEGATIVE_SLOPE(input, m)
#elif defined ACTIVATION_FUNCTION_BRELU
    #define ACTIVATION(input, m, n) ACTIVATION_BRELU(input, m)
#elif defined ACTIVATION_FUNCTION_SOFTRELU
    #define ACTIVATION(input, m, n) ACTIVATION_SOFTRELU(input)    
#elif defined ACTIVATION_FUNCTION_ABS
    #define ACTIVATION(input, m, n) ACTIVATION_ABS(input)
#elif defined ACTIVATION_FUNCTION_LINEAR
    #define ACTIVATION(input, m, n) ACTIVATION_LINEAR(input, m, n)
#elif defined ACTIVATION_FUNCTION_SQUARE
    #define ACTIVATION(input, m, n) ACTIVATION_SQUARE(input)
#elif defined ACTIVATION_FUNCTION_SQRT
    #define ACTIVATION(input, m, n) ACTIVATION_SQRT(input)
#else
    #define ACTIVATION(input, m, n) input
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define __CAT_FUNC(x, y) FUNC(x##y)
#define CAT_FUNC(x, y) __CAT_FUNC(x, y)

#define __CAT_FUNC_CALL(x, y) FUNC_CALL(x##y)
#define CAT_FUNC_CALL(x, y) __CAT_FUNC_CALL(x, y)

#define LOOP0(VAR, STMT) 
#define LOOP1(VAR, STMT) (STMT); (VAR)++;
#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;
#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;
#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;
#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;
#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;
#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;
#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;
#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;
#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;
#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;
#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;
#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;
#define LOOP14(VAR, STMT) LOOP13(VAR, STMT); (STMT); (VAR)++;
#define LOOP15(VAR, STMT) LOOP14(VAR, STMT); (STMT); (VAR)++;
#define LOOP16(VAR, STMT) LOOP15(VAR, STMT); (STMT); (VAR)++;
#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))

#define TRANSPOSE_BLOCK_8( _block )   \
        (float8)( intel_sub_group_shuffle( _block, 0 ), \
                  intel_sub_group_shuffle( _block, 1 ), \
                  intel_sub_group_shuffle( _block, 2 ), \
                  intel_sub_group_shuffle( _block, 3 ), \
                  intel_sub_group_shuffle( _block, 4 ), \
                  intel_sub_group_shuffle( _block, 5 ), \
                  intel_sub_group_shuffle( _block, 6 ), \
                  intel_sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_FP16( _block )   \
        (half8)( intel_sub_group_shuffle( _block, 0 ), \
                  intel_sub_group_shuffle( _block, 1 ), \
                  intel_sub_group_shuffle( _block, 2 ), \
                  intel_sub_group_shuffle( _block, 3 ), \
                  intel_sub_group_shuffle( _block, 4 ), \
                  intel_sub_group_shuffle( _block, 5 ), \
                  intel_sub_group_shuffle( _block, 6 ), \
                  intel_sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_COL( _block, _col )   \
        (float8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_8_COL_FP16( _block, _col )   \
        (half8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_16_FP16(_block)  \
        (half16)(as_half2(intel_sub_group_shuffle(_block, 0)),  \
                 as_half2(intel_sub_group_shuffle(_block, 1)),  \
                 as_half2(intel_sub_group_shuffle(_block, 2)),  \
                 as_half2(intel_sub_group_shuffle(_block, 3)),  \
                 as_half2(intel_sub_group_shuffle(_block, 4)),  \
                 as_half2(intel_sub_group_shuffle(_block, 5)),  \
                 as_half2(intel_sub_group_shuffle(_block, 6)),  \
                 as_half2(intel_sub_group_shuffle(_block, 7)));

#define TRANSPOSE_BLOCK_16_FP16_HALF_TYPE(_block)  \
        (half16)(intel_sub_group_shuffle(_block, 0),  \
                 intel_sub_group_shuffle(_block, 1),  \
                 intel_sub_group_shuffle(_block, 2),  \
                 intel_sub_group_shuffle(_block, 3),  \
                 intel_sub_group_shuffle(_block, 4),  \
                 intel_sub_group_shuffle(_block, 5),  \
                 intel_sub_group_shuffle(_block, 6),  \
                 intel_sub_group_shuffle(_block, 7),  \
                 intel_sub_group_shuffle(_block, 8),  \
                 intel_sub_group_shuffle(_block, 9),  \
                 intel_sub_group_shuffle(_block, 10),  \
                 intel_sub_group_shuffle(_block, 11),  \
                 intel_sub_group_shuffle(_block, 12),  \
                 intel_sub_group_shuffle(_block, 13),  \
                 intel_sub_group_shuffle(_block, 14),  \
                 intel_sub_group_shuffle(_block, 15));

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
{   \
        _result.s0 = mad( _rowA, intel_sub_group_shuffle( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, intel_sub_group_shuffle( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, intel_sub_group_shuffle( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, intel_sub_group_shuffle( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, intel_sub_group_shuffle( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, intel_sub_group_shuffle( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, intel_sub_group_shuffle( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, intel_sub_group_shuffle( colB, 7 ), _result.s7 );  \
}

#define ADD_BIAS_8( _result, _biasVal) \
{ \
    _result.s0 += intel_sub_group_shuffle( _biasVal, 0 ); \
    _result.s1 += intel_sub_group_shuffle( _biasVal, 1 ); \
    _result.s2 += intel_sub_group_shuffle( _biasVal, 2 ); \
    _result.s3 += intel_sub_group_shuffle( _biasVal, 3 ); \
    _result.s4 += intel_sub_group_shuffle( _biasVal, 4 ); \
    _result.s5 += intel_sub_group_shuffle( _biasVal, 5 ); \
    _result.s6 += intel_sub_group_shuffle( _biasVal, 6 ); \
    _result.s7 += intel_sub_group_shuffle( _biasVal, 7 ); \
}

#define ADD_BIAS_16_FP16( _result, _biasVal) \
{ \
    _result.s01 += as_half2(intel_sub_group_shuffle(_biasVal, 0)); \
    _result.s23 += as_half2(intel_sub_group_shuffle(_biasVal, 1)); \
    _result.s45 += as_half2(intel_sub_group_shuffle(_biasVal, 2)); \
    _result.s67 += as_half2(intel_sub_group_shuffle(_biasVal, 3)); \
    _result.s89 += as_half2(intel_sub_group_shuffle(_biasVal, 4)); \
    _result.sab += as_half2(intel_sub_group_shuffle(_biasVal, 5)); \
    _result.scd += as_half2(intel_sub_group_shuffle(_biasVal, 6)); \
    _result.sef += as_half2(intel_sub_group_shuffle(_biasVal, 7)); \
}

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + byte_offset))

#define MULTIPLY_OFFSET(elem_type, byte_offset) (byte_offset * sizeof(elem_type))