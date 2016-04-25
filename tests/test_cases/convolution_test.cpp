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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "api/neural.h"
#include "tests/gtest/gtest.h"
#include "test_utils/test_utils.h"

using namespace neural;
using namespace tests;

TEST(convolution_f32_fw, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {
//  Filter : 2x2
//  Stride : 2x2
//  Input  : 4x4
//  Output : 2x2
//
//  Input:
//  -0.5   1.5   0.5  0.5
//   1    -0.5   0.5  2
//   0.5   0    -1    1.5
//   2    -1     1   -0.5
//
//  Filter
//  -2   3.5
//   0.5 1.5
//
//  Bias
//  2
//
//  Output:
//  8   6
//  0.5 9

    auto input  = memory::create({engine::reference, memory::format::yxfb_f32, {1, {4, 4}, 1}, true});
    auto output = memory::create({engine::reference, memory::format::yxfb_f32, {1, {2, 2}, 1}, true});
    auto weights= memory::create({engine::reference, memory::format::yxfb_f32, {1, {2, 2}, 1}, true});
    auto biases = memory::create({engine::reference, memory::format::   x_f32, {1, {{1}} , 1}, true});

    set_values(input  , {-0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f});
    set_values(weights, {-2.0f, 0.5f, 3.5f, 1.5f});
    set_values(biases , {2.0f});

    auto conv = convolution::create({engine::reference, output, input, {1, {2, 2}, 1}, weights, biases, padding::zero});

    execute({conv});

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(8.0f, output_memory.get_value<float>(0));
    EXPECT_FLOAT_EQ(0.5f, output_memory.get_value<float>(1));
    EXPECT_FLOAT_EQ(6.0f, output_memory.get_value<float>(2));
    EXPECT_FLOAT_EQ(9.0f, output_memory.get_value<float>(3));
}

TEST(convolution_f32_fw, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x2x1x2
    //  Output : 1x1x1x2
    //
    //  Input:
    //  0.5  1.5    2.3 -0.4
    //  2.0  4.0    1.0  3.0
    //
    //  Filter:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias:
    //  -1
    //
    //  Output:
    //  3.65 -5.36
    //

    auto input   = memory::create({ engine::reference, memory::format::yxfb_f32,{2, {2, 2}, 1}, true });
    auto output  = memory::create({ engine::reference, memory::format::yxfb_f32,{2, {1, 1}, 1}, true });
    auto weights = memory::create({ engine::reference, memory::format::yxfb_f32,{1, {2, 2}, 1}, true });
    auto biases  = memory::create({ engine::reference, memory::format::   x_f32,{1, {{1}} , 1}, true });

    set_values(input,  { 0.5f, 2.3f, 1.5f, -0.4f, 2.0f, 1.0f, -4.0f, 3.0f });
    set_values(weights,{ -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases, { -1.0f });

    auto conv = convolution::create({ engine::reference, output, input, { 1, {2, 2}, 1 }, weights, biases, padding::zero });

    execute({ conv });

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(3.65f, output_memory.get_value<float>(0));
    EXPECT_FLOAT_EQ(-5.36f, output_memory.get_value<float>(1));
}

TEST(convolution_f32_fw, DISABLED_basic_wsiz2x2x1x3_wstr2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2x1x3
    //  Stride : 2x2
    //  Input  : 2x2x1x1
    //  Output : 1x1x3x1
    //
    //  Input:
    //  -2.3 -0.1
    //   3.1  1.9
    //
     // Filter:
    //  -1.1  1.5    0.1  0.2    2.0  -1.0
    //   0.5 -0.5    0.4  0.7    2.5  -1.5
    //
    //  Bias:
    //  0.1 -0.2 0.3
    //
    //  Output:
    //     0.7
    //   2.12
    // 3.08

    auto input   = memory::create({ engine::reference, memory::format::yxfb_f32,{1 ,{2, 2}, 1}, true });
    auto output  = memory::create({ engine::reference, memory::format::yxfb_f32,{1 ,{1, 1}, 3}, true });
    auto weights = memory::create({ engine::reference, memory::format::yxfb_f32,{3 ,{2, 2}, 1}, true });
    auto biases  = memory::create({ engine::reference, memory::format::   x_f32,{1 ,{{3 }}, 1}, true });

    set_values(input,   { -2.3f, -0.1f, 3.1f, 1.9f });
    set_values(weights, { -1.1f, 0.1f, 2.0f, 1.5f, 0.2f, -1.0f, 0.5f, 0.4f, 2.5f, -0.5f, 0.7f, -1.5f });
    set_values(biases,  { 0.1f, -0.2f, 0.3f });

    auto conv = convolution::create({ engine::reference, output, input, { 1, {2, 2}, 1 }, weights, biases, padding::zero });

    execute({ conv });

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(3.08f, output_memory.get_value<float>(0));
    EXPECT_FLOAT_EQ(2.12f, output_memory.get_value<float>(1));
    EXPECT_FLOAT_EQ(0.7f, output_memory.get_value<float>(2));
}

TEST(convolution_f32_fw, wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
//  Filter  : 3x3
//  Stride  : 2x2
//  Input   : 2x2
//  Output  : 1x1
//  Padding : zero
//
//  Input:
//  -0.5   1.0   padd
//   0.5   2.0   padd
//  padd  padd   padd
//
//  Filter
//  -2    0.5  3.5
//   1.5  4   -5
//   0.5  1.5 -1.5
//
//  Bias
//  2
//
//  Output:
//  12.25
    auto input  = memory::create({engine::reference, memory::format::yxfb_f32, { 1, {2, 2}, 1}, true});
    auto output = memory::create({engine::reference, memory::format::yxfb_f32, { 1, {1, 1}, 1}, true});
    auto weights= memory::create({engine::reference, memory::format::yxfb_f32, { 1, {3, 3}, 1}, true});
    auto biases = memory::create({engine::reference, memory::format::   x_f32, { 1, {{1}} , 1}, true});

    set_values(input  , {-0.5f, 1.0f, 0.5f, 2.0f});
    set_values(weights, {-2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f});
    set_values(biases , {2.0f});

    auto conv = convolution::create({engine::reference, output, input, {1, {2, 2}, 1}, weights, biases, padding::zero});
    execute({conv});

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(12.25f, output_memory.get_value<float>(0));
}

TEST(convolution_f32_fw, offsets_wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
//   Filter       : 3x3
//   Stride       : 2x2
//   Input        : 2x2
//   Input offset : -1x-1
//   Output       : 2x2
//   Output offset: 1x1
//   Padding      : zero
//
//   Input:
//   padd padd  padd
//   padd -0.5   1
//   padd  0.5   2.0
//
//   Filter
//   -2    0.5  3.5
//    1.5  4   -5
//    0.5  1.5 -1.5
//
//   Bias
//   2
//
//   Output:
//   rnd   rnd
//   rnd  -7.25
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {1 ,{2, 2}, 1}, true});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {1 ,{2, 2}, 1}, true});
    auto weights= memory::create({engine::cpu, memory::format::yxfb_f32, {1 ,{3, 3}, 1}, true});
    auto biases = memory::create({engine::cpu, memory::format::   x_f32, {1 ,{{1}} , 1}, true});

    set_values(input  , {-0.5f, 1.0f, 0.5f, 2.0f});
    set_values(weights, {-2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f});
    set_values(biases , {2.0f});

    auto conv = convolution::create({engine::reference,
                                     output,
                                     {0, {1, 1}, 0},
                                     {1, {1, 1}, 1},
                                     input,
                                     {0, {-1, -1}, 0},
                                     {1, { 2,  2}, 1},
                                     weights,
                                     biases,
                                     padding::zero});
    execute({conv});

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(-7.25f, output_memory.get_value<float>(3));
}
/*

TEST(convolution_f32_bw, wsiz2x2_wstr1x1_in2x2x1x1_nopad) {
//   Filter    : 2x2
//   Stride    : 1x1
//   FW Input  : 3x3
//   BW Input  : 2x2
//   BW Output : 3x3
//
//   FW Input:
//   -0.5   1.5  1
//    1    -0.5  2
//    1     2    3
//
//   BW Input
//   1   2
//   3   4
//
//   Filter
//   -2   3.5
//    0.5 1.5
//
//   Bias
//   2
//
//   BW Output:
//   -2   -0.5   7
//   -5.5  5    17
//    1.5  6.5   6
//
//   Weights grad
//   -7    35
//    5.5  32.25
//
//   Bias grad
//   10
    auto bw_output    = memory::create({engine::reference, memory::format::yxfb_f32, {1 ,{3, 3}, 1}, true});
    auto bw_input     = memory::create({engine::reference, memory::format::yxfb_f32, {1 ,{2, 2}, 1}, true});
    auto fw_input     = memory::create({engine::reference, memory::format::yxfb_f32, {1 ,{3, 3}, 1}, true});
    auto weights      = memory::create({engine::reference, memory::format::yxfb_f32, {1 ,{2, 2}, 1}, true});
    auto weights_diff = memory::create({engine::reference, memory::format::yxfb_f32, {1 ,{2, 2}, 1}, true});
    auto biases       = memory::create({engine::reference, memory::format::x_f32,    {1 ,{{1}} , 1}, true});
    auto biases_diff  = memory::create({engine::reference, memory::format::x_f32,    {1 ,{{1}} , 1}, true});

    auto& bw_output_mem    = bw_output.as<const memory&>();
    auto& bw_input_mem     = bw_input.as<const memory&>();
    auto& fw_input_mem     = fw_input.as<const memory&>();
    auto& weights_mem      = weights.as<const memory&>();
    auto& weights_diff_mem = weights_diff.as<const memory&>();
    auto& biases_mem       = biases.as<const memory&>();
    auto& biases_diff_mem  = biases_diff.as<const memory&>();

    set_values( fw_input, {-0.5f, 1.5f, 1.0f, 1.0f, -0.5f, 2.0f, 1.0f, 2.0f, 3.0f});
    set_values( bw_input, {1.0f, 2.0f, 3.0f, 4.0f});
    set_values( weights , {-2.0f, 3.5f, 0.5f, 1.5f});
    set_values( biases  , {2.0f});

    auto conv_bw = convolution_backward::create({engine::reference,
                                                 std::vector<primitive>{bw_output, weights_diff, biases_diff},
                                                 {bw_input, fw_input, weights, biases},
                                                 {1, {1, 1}, 1},
                                                 padding::zero});

    execute({conv_bw});

    bool results_equal = true;
    results_equal &= -2.0f == bw_output_mem.get_value<float>(0);
    results_equal &= -0.5f == bw_output_mem.get_value<float>(1);
    results_equal &=  7.0f == bw_output_mem.get_value<float>(2);
    results_equal &= -5.5f == bw_output_mem.get_value<float>(3);
    results_equal &=  5.0f == bw_output_mem.get_value<float>(4);
    results_equal &= 17.0f == bw_output_mem.get_value<float>(5);
    results_equal &=  1.5f == bw_output_mem.get_value<float>(6);
    results_equal &=  6.5f == bw_output_mem.get_value<float>(7);
    results_equal &=  6.0f == bw_output_mem.get_value<float>(8);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong output gradient";

    results_equal = true;
    results_equal &= -7.00f == weights_diff_mem.get_value<float>(0);
    results_equal &= 35.00f == weights_diff_mem.get_value<float>(1);
    results_equal &=  5.50f == weights_diff_mem.get_value<float>(2);
    results_equal &= 32.25f == weights_diff_mem.get_value<float>(3);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong weights gradient";

    results_equal = true;
    results_equal &= 10.0f == biases_diff_mem.get_value<float>(0);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong bias gradient";
}
/*

TEST(convolution_f32_bw, wsiz3x3_wstr2x2_in1x1x1x1_zeropad) {
//  Filter    : 3x3
//  Stride    : 2x2
//  FW Input  : 2x2
//  BW Input  : 1x1
//  BW Output : 2x2
//
//  FW Input:
//  -0.5   1.5  padd
//   1    -0.5  padd
//   padd padd  padd
//
//  BW Input
//  2
//
//  Filter
//  -2   3.5  1
//   0.5 1.5  2
//   1   2    3
//
//  Bias
//  -3
//
//  BW Output:
//   -4    7    padd
//    1    3    padd
//  padd  padd  padd
//
//  Weights grad
//   2  10.5  0
//   1  -1.5  0
//   0   0    0
//
//  Bias grad
//  -3
    using namespace neural;
    auto bw_output    = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto bw_input     = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {1, 1, 1, 1}, true});
    auto fw_input     = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto weights      = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto weights_diff = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto biases       = memory_obselote::create({engine::cpu, memory_obselote::format::x_f32,    {1}         , true});
    auto biases_diff  = memory_obselote::create({engine::cpu, memory_obselote::format::x_f32,    {1}         , true});

    auto& bw_output_mem    = bw_output.as<const memory_obselote&>();
    auto& bw_input_mem     = bw_input.as<const memory_obselote&>();
    auto& fw_input_mem     = fw_input.as<const memory_obselote&>();
    auto& weights_mem      = weights.as<const memory_obselote&>();
    auto& weights_diff_mem = weights_diff.as<const memory_obselote&>();
    auto& biases_mem       = biases.as<const memory_obselote&>();
    auto& biases_diff_mem  = biases_diff.as<const memory_obselote&>();

    set_values_obsolete( fw_input, {-0.5f, 1.5f, 1.0f,-0.5f});
    set_values_obsolete( bw_input, {2.0f});
    set_values_obsolete( weights , {-2.0f, 3.5f, 1.0f, 0.5f, 1.5f, 2.0f, 1.0f, 2.0f, 3.0f});
    set_values_obsolete( biases  , {-3.0f});

    auto conv_bw = convolution_backward::create({engine::reference,
                                                 std::vector<primitive>{bw_output, weights_diff, biases_diff},
                                                 {bw_input, fw_input, weights, biases},
                                                 {1, 1, 1, 1},
                                                 padding::zero});
    execute({conv_bw});

    bool results_equal = true;
    results_equal &= -4.0f == bw_output_mem.get_value<float>(0);
    results_equal &=  7.0f == bw_output_mem.get_value<float>(1);
    results_equal &=  1.0f == bw_output_mem.get_value<float>(2);
    results_equal &=  3.0f == bw_output_mem.get_value<float>(3);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong output gradient";

    results_equal = true;
    results_equal &=  2.0f == weights_diff_mem.get_value<float>(0);
    results_equal &= 10.5f == weights_diff_mem.get_value<float>(1);
    results_equal &=  0.0f == weights_diff_mem.get_value<float>(2);
    results_equal &=  1.0f == weights_diff_mem.get_value<float>(3);
    results_equal &= -1.5f == weights_diff_mem.get_value<float>(4);
    results_equal &=  0.0f == weights_diff_mem.get_value<float>(5);
    results_equal &=  0.0f == weights_diff_mem.get_value<float>(6);
    results_equal &=  0.0f == weights_diff_mem.get_value<float>(7);
    results_equal &=  0.0f == weights_diff_mem.get_value<float>(8);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong weights gradient";

    results_equal = true;
    results_equal &= 2.0f == biases_diff_mem.get_value<float>(0);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong bias gradient";
}
/*

TEST(convolution_f32_bw, offsets_wsiz3x3_in2x2x1x1_zeropad) {
//  Filter      : 3x3
//  Stride      : 1x1
//  FW Input    : 4x4
//  BW Input    : 2x2
//  BWin offset : 1x1
//  BW Output   : 4x4
//  BWout offset: 1x1 (the same offset applies to FWin)
//
//  FW Input:
//  1   1   1     1
//  1   1   1     1
//  1   1  -0.5   1.5
//  1   1   1    -0.5
//
//  BW Input
//  1  1
//  1  2
//
//  Filter
//  1   1    1
//  1  -2    3.5
//  1   0.5  1.5
//
//  Bias
//  -3
//
//  BW Output:
//  0   0   0   0
//  0   2   2   2
//  0   2  -4   7
//  0   2   1   3
//
//
//  Weights grad
//  2   2   2
//  2   2  10.5
//  2   1  -1.5
//
//  Bias grad
//  -3
    using namespace neural;
    auto bw_output    = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {4, 4, 1, 1}, true});
    auto bw_input     = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto fw_input     = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {4, 4, 1, 1}, true});
    auto weights      = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto weights_diff = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto biases       = memory_obselote::create({engine::cpu, memory_obselote::format::x_f32,    {1}         , true});
    auto biases_diff  = memory_obselote::create({engine::cpu, memory_obselote::format::x_f32,    {1}         , true});

    auto& bw_output_mem    = bw_output.as<const memory_obselote&>();
    auto& bw_input_mem     = bw_input.as<const memory_obselote&>();
    auto& fw_input_mem     = fw_input.as<const memory_obselote&>();
    auto& weights_mem      = weights.as<const memory_obselote&>();
    auto& weights_diff_mem = weights_diff.as<const memory_obselote&>();
    auto& biases_mem       = biases.as<const memory_obselote&>();
    auto& biases_diff_mem  = biases_diff.as<const memory_obselote&>();

    fw_input_mem.fill(1.0f);
    static_cast<float*>(fw_input_mem.pointer)[10] = -0.5f;
    static_cast<float*>(fw_input_mem.pointer)[11] =  1.5f;
    static_cast<float*>(fw_input_mem.pointer)[14] =  1.0f;
    static_cast<float*>(fw_input_mem.pointer)[15] = -0.5f;

    bw_input_mem.fill(1.0f);
    static_cast<float*>(bw_input_mem.pointer)[3] = 2.0f;

    weights_mem.fill(1.0f);
    static_cast<float*>(weights_mem.pointer)[4] = -2.0f;
    static_cast<float*>(weights_mem.pointer)[5] =  3.5f;
    static_cast<float*>(weights_mem.pointer)[7] =  0.5f;
    static_cast<float*>(weights_mem.pointer)[8] =  1.5f;

    biases_mem.fill(-3.0f);

    auto conv_bw = convolution_backward::create({engine::reference,
                                                 std::vector<primitive>{bw_output, weights_diff, biases_diff},
                                                 {1, 1, 0, 0},
                                                 {1, 1, 1, 1},
                                                 {bw_input, fw_input, weights, biases},
                                                 {1, 1, 0, 0},
                                                 {1, 1, 1, 1},
                                                 padding::zero});
    execute({conv_bw});

    bool results_equal = true;
    results_equal &=  0.0f == bw_output_mem.get_value<float>(0);
    results_equal &=  0.0f == bw_output_mem.get_value<float>(1);
    results_equal &=  0.0f == bw_output_mem.get_value<float>(2);
    results_equal &=  0.0f == bw_output_mem.get_value<float>(3);
    results_equal &=  0.0f == bw_output_mem.get_value<float>(4);
    results_equal &=  2.0f == bw_output_mem.get_value<float>(5);
    results_equal &=  2.0f == bw_output_mem.get_value<float>(6);
    results_equal &=  2.0f == bw_output_mem.get_value<float>(7);
    results_equal &=  0.0f == bw_output_mem.get_value<float>(8);
    results_equal &=  2.0f == bw_output_mem.get_value<float>(9);
    results_equal &= -4.0f == bw_output_mem.get_value<float>(10);
    results_equal &=  7.0f == bw_output_mem.get_value<float>(11);
    results_equal &=  0.0f == bw_output_mem.get_value<float>(12);
    results_equal &=  2.0f == bw_output_mem.get_value<float>(13);
    results_equal &=  1.0f == bw_output_mem.get_value<float>(14);
    results_equal &=  3.0f == bw_output_mem.get_value<float>(15);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong output gradient";

    results_equal = true;
    results_equal &=  2.0f == weights_diff_mem.get_value<float>(0);
    results_equal &=  2.0f == weights_diff_mem.get_value<float>(1);
    results_equal &=  2.0f == weights_diff_mem.get_value<float>(2);
    results_equal &=  2.0f == weights_diff_mem.get_value<float>(3);
    results_equal &=  2.0f == weights_diff_mem.get_value<float>(4);
    results_equal &= 10.5f == weights_diff_mem.get_value<float>(5);
    results_equal &=  2.0f == weights_diff_mem.get_value<float>(6);
    results_equal &=  1.0f == weights_diff_mem.get_value<float>(7);
    results_equal &= -1.5f == weights_diff_mem.get_value<float>(8);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong weights gradient";

    results_equal = true;
    results_equal &= 2.0f == biases_diff_mem.get_value<float>(0);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong bias gradient";
}
*/