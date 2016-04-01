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

#include "tests/gtest/gtest.h"
#include "api/neural.h"
/*

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
    using namespace neural;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {4, 4, 1, 1}, true});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto weights= memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto biases = memory::create({engine::cpu, memory::format::   x_f32, {1}         , true});

    auto& input_memory  = input.as<const memory&>();
    auto& output_memory = output.as<const memory&>();
    auto& weight_memory = weights.as<const memory&>();

    *static_cast<float*>(biases.as<const memory&>().pointer) = 2;

    input_memory.set_value(0, -0.5f);
    input_memory.set_value(1,  1.0f);
    input_memory.set_value(2,  0.5f);
    input_memory.set_value(3,  2.0f);
    input_memory.set_value(4,  1.5f);
    input_memory.set_value(5, -0.5f);
    input_memory.set_value(6,  0.0f);
    input_memory.set_value(7, -1.0f);
    input_memory.set_value(8,  0.5f);
    input_memory.set_value(9,  0.5f);
    input_memory.set_value(10, -1.0f);
    input_memory.set_value(11,  1.0f);
    input_memory.set_value(12,  0.5f);
    input_memory.set_value(13,  2.0f);
    input_memory.set_value(14,  1.5f);
    input_memory.set_value(15, -0.5f);

    weight_memory.set_value(0, -2.0f);
    weight_memory.set_value(1,  0.5f);
    weight_memory.set_value(2,  3.5f);
    weight_memory.set_value(3,  1.5f);

    auto conv = convolution::create({engine::reference, output, input, {2, 2, 1, 1}, weights, biases, padding::zero});

    execute({conv});

    ASSERT_EQ(8.0f, output_memory.get_value<float>(0));
    ASSERT_EQ(0.5f, output_memory.get_value<float>(1));
    ASSERT_EQ(6.0f, output_memory.get_value<float>(2));
    ASSERT_EQ(9.0f, output_memory.get_value<float>(3));
}

TEST(convolution_f32_fw, wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
//  Filter  : 3x3
//  Stride  : 2x2
//  Input   : 2x2
//  Output  : 1x1
//  Padding : zero
//
//  Input:
//  -0.5   0.5   padd
//   1     2.0   padd
//  padd  padd   padd
//
//  Filter
//  -2    1.5  0.5
//   0.5  4    1.5
//   3.5 -5   -1.5
//
//  Bias
//  2
//
//  Output:
//  12.25
    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {1, 1, 1, 1}, true});
    auto weights= memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto biases = memory::create({engine::cpu, memory::format::   x_f32, {1}         , true});

    auto& input_memory  = input.as<const memory&>();
    auto& output_memory = output.as<const memory&>();
    auto& weight_memory = weights.as<const memory&>();

    *static_cast<float*>(biases.as<const memory&>().pointer) = 2;

    input_memory.set_value(0, -0.5f);
    input_memory.set_value(1,  1.0f);
    input_memory.set_value(2,  0.5f);
    input_memory.set_value(3,  2.0f);

    weight_memory.set_value(0, -2.0f);
    weight_memory.set_value(1,  0.5f);
    weight_memory.set_value(2,  3.5f);
    weight_memory.set_value(3,  1.5f);
    weight_memory.set_value(4,  4.0f);
    weight_memory.set_value(5, -5.0f);
    weight_memory.set_value(6,  0.5f);
    weight_memory.set_value(7,  1.5f);
    weight_memory.set_value(8, -1.5f);

    float* inptr  = (float*)input_memory.pointer;
    float* wptr   = (float*)weight_memory.pointer;
    float* outptr = (float*)output_memory.pointer;

    auto conv = convolution::create({engine::reference, output, input, {2, 2, 1, 1}, weights, biases, padding::zero});
    execute({conv});

    ASSERT_EQ(12.25f, output_memory.get_value<float>(0));
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
//   padd -0.5   0.5
//   padd  1     2.0
//
//   Filter
//   -2    1.5  0.5
//    0.5  4    1.5
//    3.5 -5   -1.5
//
//   Bias
//   2
//
//   Output:
//   rnd   rnd
//   rnd  -7.25
    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto weights= memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto biases = memory::create({engine::cpu, memory::format::   x_f32, {1}         , true});

    auto& input_memory  = input.as<const memory&>();
    auto& output_memory = output.as<const memory&>();
    auto& weight_memory = weights.as<const memory&>();

    *static_cast<float*>(biases.as<const memory&>().pointer) = 2;

    input_memory.set_value(0, -0.5f);
    input_memory.set_value(1,  1.0f);
    input_memory.set_value(2,  0.5f);
    input_memory.set_value(3,  2.0f);

    weight_memory.set_value(0, -2.0f);
    weight_memory.set_value(1,  0.5f);
    weight_memory.set_value(2,  3.5f);
    weight_memory.set_value(3,  1.5f);
    weight_memory.set_value(4,  4.0f);
    weight_memory.set_value(5, -5.0f);
    weight_memory.set_value(6,  0.5f);
    weight_memory.set_value(7,  1.5f);
    weight_memory.set_value(8, -1.5f);

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

    ASSERT_EQ(-7.25f, output_memory.get_value<float>(3));
}

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
//   -3
//
//   BW Output:
//   -2   -0.5   7
//   -5.5  5    17
//    1.5  6.5   6
//
//   Weights grad
//   -7    35
//    5.5  32.5
//
//   Bias grad
//   10
    using namespace neural;
    auto bw_output    = memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto bw_input     = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto fw_input     = memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto weights      = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto weights_diff = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto biases       = memory::create({engine::cpu, memory::format::x_f32,    {1}         , true});
    auto biases_diff  = memory::create({engine::cpu, memory::format::x_f32,    {1}         , true});

    auto& bw_output_mem    = bw_output.as<const memory&>();
    auto& bw_input_mem     = bw_input.as<const memory&>();
    auto& fw_input_mem     = fw_input.as<const memory&>();
    auto& weights_mem      = weights.as<const memory&>();
    auto& weights_diff_mem = weights_diff.as<const memory&>();
    auto& biases_mem       = biases.as<const memory&>();
    auto& biases_diff_mem  = biases_diff.as<const memory&>();

    fw_input_mem.set_value(0, -0.5f);
    fw_input_mem.set_value(1,  1.5f);
    fw_input_mem.set_value(2,  1.0f);
    fw_input_mem.set_value(3,  1.0f);
    fw_input_mem.set_value(4, -0.5f);
    fw_input_mem.set_value(5,  2.0f);
    fw_input_mem.set_value(6,  1.0f);
    fw_input_mem.set_value(7,  2.0f);
    fw_input_mem.set_value(8,  3.0f);

    bw_input_mem.set_value(0, 1.0f);
    bw_input_mem.set_value(1, 2.0f);
    bw_input_mem.set_value(2, 3.0f);
    bw_input_mem.set_value(3, 4.0f);

    weights_mem.set_value(0, -2.0f);
    weights_mem.set_value(1,  3.5f);
    weights_mem.set_value(2,  0.5f);
    weights_mem.set_value(3,  1.5f);

    biases_mem.set_value(0, -3.0f);


    auto conv_bw = convolution_backward::create({engine::reference,
                                                 std::vector<primitive>{bw_output, weights_diff, biases_diff},
                                                 {bw_input, fw_input, weights, biases},
                                                 {1, 1, 1, 1},
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
    EXPECT_EQ(true, results_equal) << "ERROR MESSAGE: wrong output gradient";

    results_equal = true;
    results_equal &= -7.00f == weights_diff_mem.get_value<float>(0);
    results_equal &= 35.00f == weights_diff_mem.get_value<float>(1);
    results_equal &=  5.50f == weights_diff_mem.get_value<float>(2);
    results_equal &= 32.25f == weights_diff_mem.get_value<float>(3);
    EXPECT_EQ(true, results_equal) << "ERROR MESSAGE: wrong weights gradient";

    results_equal = true;
    results_equal &= 10.0f == biases_diff_mem.get_value<float>(0);
    EXPECT_EQ(true, results_equal) << "ERROR MESSAGE: wrong bias gradient";
}

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
//  2
    using namespace neural;
    auto bw_output    = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto bw_input     = memory::create({engine::cpu, memory::format::yxfb_f32, {1, 1, 1, 1}, true});
    auto fw_input     = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto weights      = memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto weights_diff = memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto biases       = memory::create({engine::cpu, memory::format::x_f32,    {1}         , true});
    auto biases_diff  = memory::create({engine::cpu, memory::format::x_f32,    {1}         , true});

    auto& bw_output_mem    = bw_output.as<const memory&>();
    auto& bw_input_mem     = bw_input.as<const memory&>();
    auto& fw_input_mem     = fw_input.as<const memory&>();
    auto& weights_mem      = weights.as<const memory&>();
    auto& weights_diff_mem = weights_diff.as<const memory&>();
    auto& biases_mem       = biases.as<const memory&>();
    auto& biases_diff_mem  = biases_diff.as<const memory&>();

    fw_input_mem.set_value(0, -0.5f);
    fw_input_mem.set_value(1,  1.5f);
    fw_input_mem.set_value(2,  1.0f);
    fw_input_mem.set_value(3, -0.5f);

    bw_input_mem.set_value(0, 2.0f);

    weights_mem.set_value(0, -2.0f);
    weights_mem.set_value(1,  3.5f);
    weights_mem.set_value(2,  1.0f);
    weights_mem.set_value(3,  0.5f);
    weights_mem.set_value(4,  1.5f);
    weights_mem.set_value(5,  2.0f);
    weights_mem.set_value(6,  1.0f);
    weights_mem.set_value(7,  2.0f);
    weights_mem.set_value(8,  3.0f);

    biases_mem.set_value(0, -3.0f);

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
    EXPECT_EQ(true, results_equal) << "ERROR MESSAGE: wrong output gradient";

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
    EXPECT_EQ(true, results_equal) << "ERROR MESSAGE: wrong weights gradient";

    results_equal = true;
    results_equal &= 2.0f == biases_diff_mem.get_value<float>(0);
    EXPECT_EQ(true, results_equal) << "ERROR MESSAGE: wrong bias gradient";
}

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
//  2
    using namespace neural;
    auto bw_output    = memory::create({engine::cpu, memory::format::yxfb_f32, {4, 4, 1, 1}, true});
    auto bw_input     = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto fw_input     = memory::create({engine::cpu, memory::format::yxfb_f32, {4, 4, 1, 1}, true});
    auto weights      = memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto weights_diff = memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto biases       = memory::create({engine::cpu, memory::format::x_f32,    {1}         , true});
    auto biases_diff  = memory::create({engine::cpu, memory::format::x_f32,    {1}         , true});

    auto& bw_output_mem    = bw_output.as<const memory&>();
    auto& bw_input_mem     = bw_input.as<const memory&>();
    auto& fw_input_mem     = fw_input.as<const memory&>();
    auto& weights_mem      = weights.as<const memory&>();
    auto& weights_diff_mem = weights_diff.as<const memory&>();
    auto& biases_mem       = biases.as<const memory&>();
    auto& biases_diff_mem  = biases_diff.as<const memory&>();


    fw_input_mem.fill(1.0f);
    fw_input_mem.set_value(10, -0.5f);
    fw_input_mem.set_value(11,  1.5f);
    fw_input_mem.set_value(14,  1.0f);
    fw_input_mem.set_value(15, -0.5f);

    bw_input_mem.fill(1.0f);
    bw_input_mem.set_value(3, 2.0f);

    weights_mem.fill(1.0f);
    weights_mem.set_value(4, -2.0f);
    weights_mem.set_value(5,  3.5f);
    weights_mem.set_value(7,  0.5f);
    weights_mem.set_value(8,  1.5f);

    biases_mem.set_value(0, -3.0f);

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
    EXPECT_EQ(true, results_equal) << "ERROR MESSAGE: wrong output gradient";

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
    EXPECT_EQ(true, results_equal) << "ERROR MESSAGE: wrong weights gradient";

    results_equal = true;
    results_equal &= 2.0f == biases_diff_mem.get_value<float>(0);
    EXPECT_EQ(true, results_equal) << "ERROR MESSAGE: wrong bias gradient";
}
*/