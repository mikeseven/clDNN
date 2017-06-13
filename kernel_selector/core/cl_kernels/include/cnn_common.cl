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

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

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


typedef struct half1  { half s0; }                                                               half1;
typedef struct half5  { half s0; half s1; half s2; half s3; half s4; }                           half5;
typedef struct half6  { half s0; half s1; half s2; half s3; half s4; half s5; }                  half6;
typedef struct half7  { half s0; half s1; half s2; half s3; half s4; half s5; half s6; }         half7;
typedef struct half9  { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8; }                                                               half9;
typedef struct half10 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8; half s9; }                                                      half10;
typedef struct half11 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8; half s9; half sa; }                                             half11;
typedef struct half12 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8;  half s9; half sa; half sb;}                                    half12;
typedef struct half13 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8;  half s9; half sa; half sb; half sc;}                           half13;
typedef struct half14 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8;  half s9; half sa; half sb; half sc; half se;}                  half14;
typedef struct half15 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                       half s8;  half s9; half sa; half sb; half sc; half se; half sf;}          half15;
typedef struct half0  { half s0; } half0; //never used but makes compiler happy.

typedef struct short1  { short s0; }                                                                      short1;
typedef struct short5  { short s0; short s1; short s2; short s3; short s4; }                              short5;
typedef struct short6  { short s0; short s1; short s2; short s3; short s4; short s5; }                    short6;
typedef struct short7  { short s0; short s1; short s2; short s3; short s4; short s5; short s6; }          short7;
typedef struct short9  { short s0; short s1; short s2; short s3; short s4; short s5; short s6; short s7; 
                         short s8; }                                                                      short9;
typedef struct short10 { short s0; short s1; short s2; short s3; short s4; short s5; short s6; short s7; 
                         short s8; short s9; }                                                            short10;
typedef struct short11 { short s0; short s1; short s2; short s3; short s4; short s5; short s6; short s7; 
                         short s8;  short s9; short sa; }                                                 short11;
typedef struct short12 { short s0; short s1; short s2; short s3; short s4; short s5; short s6; short s7; 
                         short s8;  short s9; short sa; short sb;}                                        short12;
typedef struct short13 { short s0; short s1; short s2; short s3; short s4; short s5; short s6; short s7; 
                         short s8;  short s9; short sa; short sb; short sc;}                              short13;
typedef struct short14 { short s0; short s1; short s2; short s3; short s4; short s5; short s6; short s7; 
                         short s8;  short s9; short sa; short sb; short sc; short se;}                    short14;
typedef struct short15 { short s0; short s1; short s2; short s3; short s4; short s5; short s6; short s7; 
                         short s8;  short s9; short sa; short sb; short sc; short se; short sf;}          short15;
typedef struct short0 { short s0; } short0; //never used but makes compiler happy.

typedef struct float1 { float s0; } float1;
typedef struct float5 { float s0; float s1; float s2; float s3; float s4; } float5;
typedef struct float6 { float s0; float s1; float s2; float s3; float s4; float s5; } float6;
typedef struct float7 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; } float7;
typedef struct float9 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; float s7; float s8; } float9;
typedef struct float10 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9;} float10;
typedef struct float11 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa;} float11;
typedef struct float12 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; } float12;
typedef struct float13 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc;} float13;
typedef struct float14 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; } float14;
typedef struct float15 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; float se; } float15;
typedef struct float0 { float s0; } float0; //never used but makes compiler happy.

#if (KERNEL_WIDTH == 1)
__constant half1 half_zeros= (half1){0};
#elif (KERNEL_WIDTH == 2)
    __constant half2 half_zeros = (half2)(0);
#elif (KERNEL_WIDTH == 3)
    __constant half3 half_zeros = (half3)(0);
#elif (KERNEL_WIDTH == 4)
    __constant half4 half_zeros = (half4)(0);
#elif (KERNEL_WIDTH == 5)
    __constant half5 half_zeros = (half5){0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 6)
    __constant half6 half_zeros = (half6){0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 7)
    __constant half7 half_zeros = (half7){0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 8)
    __constant half8 half_zeros = (half8)(0);
#elif (KERNEL_WIDTH == 9)
    __constant half9 half_zeros = (half9){0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 10)
    __constant half10 half_zeros = (half10){0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 11)
    __constant half11 half_zeros = (half11){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 12)
    __constant half12 half_zeros = (half12){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 13)
    __constant half13 half_zeros = (half13){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 14)
    __constant half14 half_zeros = (half14){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 15)
    __constant half15 half_zeros = (half15){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 16)
    __constant half16 half_zeros = (half16)(0);
#endif

#if (KERNEL_WIDTH == 1)
__constant short1 short_zeros     = (short1){0};
#elif (KERNEL_WIDTH == 2)
    __constant short2 short_zeros = (short2)(0);
#elif (KERNEL_WIDTH == 3)
    __constant short3 short_zeros = (short3)(0);
#elif (KERNEL_WIDTH == 4)
    __constant short4 short_zeros = (short4)(0);
#elif (KERNEL_WIDTH == 5)
    __constant short5 short_zeros = (short5){0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 6)
    __constant short6 short_zeros = (short6){0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 7)
    __constant short7 short_zeros = (short7){0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 8)
    __constant short8 short_zeros = (short8)(0);
#elif (KERNEL_WIDTH == 9)
    __constant short9 short_zeros = (short9){0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 10)
    __constant short10 short_zeros = (short10){0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 11)
    __constant short11 short_zeros = (short11){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 12)
    __constant short12 short_zeros = (short12){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 13)
    __constant short13 short_zeros = (short13){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 14)
    __constant short14 short_zeros = (short14){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 15)
    __constant short15 short_zeros = (short15){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 16)
    __constant short16 short_zeros = (short16)(0);
#endif

#if (KERNEL_WIDTH == 1)
__constant float1 float_zeros     = (float1){0};
#elif (KERNEL_WIDTH == 2)
    __constant float2 float_zeros = (float2)(0);
#elif (KERNEL_WIDTH == 3)
    __constant float3 float_zeros = (float3)(0);
#elif (KERNEL_WIDTH == 4)
    __constant float4 float_zeros = (float4)(0);
#elif (KERNEL_WIDTH == 5)
    __constant float5 float_zeros = (float5){0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 6)
    __constant float6 float_zeros = (float6){0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 7)
    __constant float7 float_zeros = (float7){0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 8)
    __constant float8 float_zeros = (float8)(0);
#elif (KERNEL_WIDTH == 9)
    __constant float9 float_zeros = (float9){0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 10)
    __constant float10 float_zeros = (float10){0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 11)
    __constant float11 float_zeros = (float11){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 12)
    __constant float12 float_zeros = (float12){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 13)
    __constant float13 float_zeros = (float13){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 14)
    __constant float14 float_zeros = (float14){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 15)
    __constant float15 float_zeros = (float15){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 16)
    __constant float16 float_zeros = (float16)(0);
#endif

#undef COUNTER_TYPE

#if defined(COUNTER_TYPE_F16)
     #define COUNTER_TYPE half
#elif defined(COUNTER_TYPE_F32)
     #define COUNTER_TYPE float
#endif

#if defined TYPE_F16
    #define DATA_TYPE half

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    #if !defined(COUNTER_TYPE)
        #define COUNTER_TYPE_F32
        #define COUNTER_TYPE float
    #endif
    
    #define DATA_TYPE_MAX HALF_MAX
    #define DATA_TYPE_MIN -HALF_MAX
    #define DATA_TYPE_ZERO 0.0h
#elif defined TYPE_F32
    #define DATA_TYPE float

    #if !defined(COUNTER_TYPE)
        #define COUNTER_TYPE_F32
        #define COUNTER_TYPE float
    #endif
    
    #define DATA_TYPE_MAX FLT_MAX
    #define DATA_TYPE_MIN -FLT_MAX
    #define DATA_TYPE_ZERO 0.0f
#endif

#if defined ACTIVATION_FUNCTION_LOGISTIC
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return (TYPE_T)(1.0) / ((TYPE_T)(1.0) + exp(-value)); }

#elif defined ACTIVATION_FUNCTION_HYPERBOLIC_TAN
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return tanh(value); }

#elif defined ACTIVATION_FUNCTION_RELU
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return fmax(value, (TYPE_T)(0)); }

#elif defined ACTIVATION_FUNCTION_SOFTRELU
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return log( (TYPE_T)(1) + exp(value)); }
    
#elif defined ACTIVATION_FUNCTION_RELU_NEGATIVE_SLOPE
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return isinf((TYPE_T)m) ? ((value >= (TYPE_T)0) ? value : -(TYPE_T)m) : (fmax(value, (TYPE_T)0) + (TYPE_T)m * fmin(value, (TYPE_T)0)); }

#elif defined ACTIVATION_FUNCTION_ABS
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return fabs(value); }

#elif defined ACTIVATION_FUNCTION_SQUARE
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return value * value; }

#elif defined ACTIVATION_FUNCTION_SQRT
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return sqrt(value); }

#elif defined ACTIVATION_FUNCTION_BRELU
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return fmin((TYPE_T)(m), fmax((TYPE_T)(0), value)); }

#elif defined ACTIVATION_FUNCTION_LINEAR
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return (TYPE_T)(m) * value + (TYPE_T)(n); }

#else
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return value; }

#endif

ACTIVATION_FUNCTION(half)
ACTIVATION_FUNCTION(half2)
ACTIVATION_FUNCTION(half3)
ACTIVATION_FUNCTION(half4)
//ACTIVATION_FUNCTION(half5)
//ACTIVATION_FUNCTION(half6)
//ACTIVATION_FUNCTION(half7)
//ACTIVATION_FUNCTION(half8)
//ACTIVATION_FUNCTION(half9)
//ACTIVATION_FUNCTION(half10)
//ACTIVATION_FUNCTION(half11)
//ACTIVATION_FUNCTION(half12)
//ACTIVATION_FUNCTION(half13)
//ACTIVATION_FUNCTION(half14)
//ACTIVATION_FUNCTION(half15)
ACTIVATION_FUNCTION(half16)

ACTIVATION_FUNCTION(float)
ACTIVATION_FUNCTION(float2)
ACTIVATION_FUNCTION(float3)
ACTIVATION_FUNCTION(float4)
//ACTIVATION_FUNCTION(float5)
//ACTIVATION_FUNCTION(float6)
//ACTIVATION_FUNCTION(float7)
ACTIVATION_FUNCTION(float8)
//ACTIVATION_FUNCTION(float9)
//ACTIVATION_FUNCTION(float10)
//ACTIVATION_FUNCTION(float11)
//ACTIVATION_FUNCTION(float12)
//ACTIVATION_FUNCTION(float13)
//ACTIVATION_FUNCTION(float14)
//ACTIVATION_FUNCTION(float15)
//ACTIVATION_FUNCTION(float16)

inline DATA_TYPE activation_function(DATA_TYPE in_f, float m, float n)
{
    return CAT(activation_function_, DATA_TYPE)(in_f, m ,n);
}
