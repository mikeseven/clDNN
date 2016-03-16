/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "tests/gtest/gtest.h"
#include "api/neural.h"

TEST(convolution_f32_fw, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {
/*
Filter window: 2x2
Stride       : 2x2
Input size   : 4x4
Output size  : 2x2

Input:
-0.5   1.5   0.5  0.5
 1    -0.5   0.5  2
 0.5   0    -1    1.5
 2    -1     1   -0.5

Filter
-2   3.5
 0.5 1.5

Bias
2

Output:
8   6
0.5 9
*/
using namespace neural;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {4, 4, 1, 1}, true});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto weights= memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto biases = memory::create({engine::cpu, memory::format::   x_f32, {1}         , true});

    auto& input_memory  = input.as<const memory&>();
    auto& output_memory = output.as<const memory&>();
    auto& weight_memory = weights.as<const memory&>();

    *static_cast<float*>(biases.as<const memory&>().pointer) = 2;

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  1.0f);
    input_memory.set_value<float>(2,  0.5f);
    input_memory.set_value<float>(3,  2.0f);
    input_memory.set_value<float>(4,  1.5f);
    input_memory.set_value<float>(5, -0.5f);
    input_memory.set_value<float>(6,  0.0f);
    input_memory.set_value<float>(7, -1.0f);
    input_memory.set_value<float>(8,  0.5f);
    input_memory.set_value<float>(9,  0.5f);
    input_memory.set_value<float>(10, -1.0f);
    input_memory.set_value<float>(11,  1.0f);
    input_memory.set_value<float>(12,  0.5f);
    input_memory.set_value<float>(13,  2.0f);
    input_memory.set_value<float>(14,  1.5f);
    input_memory.set_value<float>(15, -0.5f);

    weight_memory.set_value<float>(0, -2.0f);
    weight_memory.set_value<float>(1,  0.5f);
    weight_memory.set_value<float>(2,  3.5f);
    weight_memory.set_value<float>(3,  1.5f);

    auto conv = convolution::create({engine::reference, output, input, {2, 2, 1, 1}, weights, biases, padding::zero});
    execute({conv});

    EXPECT_EQ(8.0f, output_memory.get_value<float>(0));
    EXPECT_EQ(0.5f, output_memory.get_value<float>(1));
    EXPECT_EQ(6.0f, output_memory.get_value<float>(2));
    EXPECT_EQ(9.0f, output_memory.get_value<float>(3));
}

TEST(convolution_f32_fw, basic_wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
/*
Filter window: 3x3
Stride       : 2x2
Input size   : 2x2
Output size  : 1x1
Padding      : zero

Input:
-0.5   0.5   padd
 1     2.0   padd
padd  padd   padd

Filter
-2    1.5  0.5
 0.5  4    1.5
 3.5 -5   -1.5

Bias
2

Output:
12.25
*/
using namespace neural;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {1, 1, 1, 1}, true});
    auto weights= memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto biases = memory::create({engine::cpu, memory::format::   x_f32, {1}         , true});

    auto& input_memory  = input.as<const memory&>();
    auto& output_memory = output.as<const memory&>();
    auto& weight_memory = weights.as<const memory&>();

    *static_cast<float*>(biases.as<const memory&>().pointer) = 2;

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  1.0f);
    input_memory.set_value<float>(2,  0.5f);
    input_memory.set_value<float>(3,  2.0f);

    weight_memory.set_value<float>(0, -2.0f);
    weight_memory.set_value<float>(1,  0.5f);
    weight_memory.set_value<float>(2,  3.5f);
    weight_memory.set_value<float>(3,  1.5f);
    weight_memory.set_value<float>(4,  4.0f);
    weight_memory.set_value<float>(5, -5.0f);
    weight_memory.set_value<float>(6,  0.5f);
    weight_memory.set_value<float>(7,  1.5f);
    weight_memory.set_value<float>(8, -1.5f);

    float* inptr  = (float*)input_memory.pointer;
    float* wptr   = (float*)weight_memory.pointer;
    float* outptr = (float*)output_memory.pointer;

    auto conv = convolution::create({engine::reference, output, input, {2, 2, 1, 1}, weights, biases, padding::zero});
    execute({conv});

    EXPECT_EQ(12.25f, output_memory.get_value<float>(0));
}

TEST(convolution_f32_fw, offsets_wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
/*
Filter window: 3x3
Stride       : 2x2
Input size   : 2x2
Input offset : -1x-1
Output size  : 2x2
Output offset: 1x1
Padding      : zero

Input:
padd padd  padd
padd -0.5   0.5
padd  1     2.0

Filter
-2    1.5  0.5
 0.5  4    1.5
 3.5 -5   -1.5

Bias
2

Output:
rnd   rnd
rnd  -7.25
*/
    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto weights= memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto biases = memory::create({engine::cpu, memory::format::   x_f32, {1}         , true});

    auto& input_memory  = input.as<const memory&>();
    auto& output_memory = output.as<const memory&>();
    auto& weight_memory = weights.as<const memory&>();

    *static_cast<float*>(biases.as<const memory&>().pointer) = 2;

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  1.0f);
    input_memory.set_value<float>(2,  0.5f);
    input_memory.set_value<float>(3,  2.0f);

    weight_memory.set_value<float>(0, -2.0f);
    weight_memory.set_value<float>(1,  0.5f);
    weight_memory.set_value<float>(2,  3.5f);
    weight_memory.set_value<float>(3,  1.5f);
    weight_memory.set_value<float>(4,  4.0f);
    weight_memory.set_value<float>(5, -5.0f);
    weight_memory.set_value<float>(6,  0.5f);
    weight_memory.set_value<float>(7,  1.5f);
    weight_memory.set_value<float>(8, -1.5f);

    auto conv = convolution::create({engine::reference,
                                     output,
                                     {1, 1, 0, 0},
                                     {1, 1, 1, 1},
                                     input,
                                     {-1, -1, 0, 0},
                                     { 2,  2, 1, 1},
                                     weights,
                                     biases,
                                     padding::zero});
    execute({conv});

    EXPECT_EQ(-7.25f, output_memory.get_value<float>(3));
}

