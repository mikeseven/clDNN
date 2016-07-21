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
#include "memory_utils.h"
#include <algorithm>
#include <thread>

#include "multidimensional_counter.h"//todo remove

using namespace neural;
using namespace tests;

TEST(convolution_f32_fw, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {
//  Filter : 2x2
//  Stride : 2x2
//  Input  : 4x4
//  Output : 2x2

//  Input:
//  -0.5   1     0.5  2
//   1.5  -0.5   0   -1
//   0.5   0.5  -1    1
//   0.5   2     1.5 -0.5
//
//  Filter
//  -2   0.5
//   3.5 1.5
//
//  Bias
//  2
//
//  Output:
//  8  0.5
//  6  9

    auto input  = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {4, 4}, 1}});
    auto output = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {2, 2}, 1}});
    auto weights= memory::allocate({engine::reference, memory::format::oiyx_f32, {1, {2, 2},{1, 1}}});
    auto biases = memory::allocate({engine::reference, memory::format::   x_f32, {1, {{1}} , 1}});

    set_values(input  , {
        -0.5f,  1.0f,  0.5f,  2.0f,
         1.5f, -0.5f,  0.0f, -1.0f,
         0.5f,  0.5f, -1.0f,  1.0f,
         0.5f,  2.0f,  1.5f, -0.5f
    });
    set_values(weights, {-2.0f, 0.5f, 3.5f, 1.5f});
    set_values(biases , {2.0f});

    auto conv = convolution::create({engine::reference, output, {input, weights, biases}, {1, {2, 2}, 1}, padding::zero});

    execute({conv}).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(8.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.5f, output_ptr[1]);
    EXPECT_FLOAT_EQ(6.0f, output_ptr[2]);
    EXPECT_FLOAT_EQ(9.0f, output_ptr[3]);
}

TEST(convolution_f32_fw, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad) {
//  Filter : 2x2
//  Stride : 2x2
//  Input  : 2x2x1x2
//  Output : 1x1x1x2
//
//  Input:
//  0.5  1.5    2.3 -0.4
//  2.0 -4.0    1.0  3.0
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
    auto input   = memory::allocate({ engine::reference, memory::format::yxfb_f32,{2, {2, 2}, 1}});
    auto output  = memory::allocate({ engine::reference, memory::format::yxfb_f32,{2, {1, 1}, 1}});
    auto weights = memory::allocate({ engine::reference, memory::format::oiyx_f32,{1, {2, 2},{1, 1}}});
    auto biases  = memory::allocate({ engine::reference, memory::format::   x_f32,{1, {{1}} , 1}});

    set_values(input,  { 0.5f, 2.3f, 1.5f, -0.4f, 2.0f, 1.0f, -4.0f, 3.0f });
    set_values(weights,{ -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases, { -1.0f });

    auto conv = convolution::create({ engine::reference, output, {input, weights, biases}, { 1, {2, 2}, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(3.65f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.36f, output_ptr[1]);
}

TEST(convolution_f32_fw, basic_ofm_wsiz2x1x2x1_in1x2x1_nopad) {
//  Filter : 1x2x1x2x1
//  Input  : 1x1x2x1
//  Output : 1x2x1x1
//
//  Input:
//  1.0    2.0
//
// Filter:
//   1.0    2.0  ofm=0
//  -1.0   -2.0  ofm=1
//
//  Bias:
//  0.1 -0.2
//
//  Output:
//   5.1  f=0
//  -5.2  f=1

    auto input   = memory::allocate({ engine::reference, memory::format::yxfb_f32,{1 ,{2, 1}, 1}});
    auto output  = memory::allocate({ engine::reference, memory::format::yxfb_f32,{1 ,{1, 1}, 2}});
    auto weights = memory::allocate({ engine::reference, memory::format::oiyx_f32,{1 ,{2, 1},{2, 1}}});
    auto biases  = memory::allocate({ engine::reference, memory::format::   x_f32,{1 ,{{2}}, 1}});

    set_values(input,   { 1.0f, 2.0f });
    set_values(weights, { 1.0f, 2.0f, -1.0f, -2.0f });
    set_values(biases,  { 0.1f, -0.2f});

    auto conv = convolution::create({ engine::reference, output, {input, weights, biases}, { 1, {5, 5}, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(5.1f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.2f, output_ptr[1]);
}

TEST(convolution_f32_fw, basic_ofm_wsiz3x2x2x1_in2x2x1_nopad) {
//  Filter : 1x3x2x2x1
//  Input  : 1x2x2x1
//  Output : 1x3x1x1
//
//  Input:
//  1.0    2.0  f=0
//  3.0    4.0  f=1
//
// Filter:
//   1.0    2.0  ifm=0  ofm=0
//   3.0    4.0  ifm=1
//
//   5.0    6.0  ifm=0  ofm=1
//   7.0    8.0  ifm=1
//
//   9.0   10.0  ifm=0  ofm=2
//  11.0   12.0  ifm=1
//  Bias:
//   -5     -6     -7
//
//  Output:
//   25.0  f=0
//   64,0  f=1
//  103.0  f=2

    auto input   = memory::allocate({ engine::reference, memory::format::yxfb_f32,{1 ,{2, 1}, 2}});
    auto output  = memory::allocate({ engine::reference, memory::format::yxfb_f32,{1 ,{1, 1}, 3}});
    auto weights = memory::allocate({ engine::reference, memory::format::oiyx_f32,{1 ,{2, 1},{3, 2}}});
    auto biases  = memory::allocate({ engine::reference, memory::format::   x_f32,{1 ,{{3}}, 1}});

    set_values(input,   { 1.0f, 3.0f, 2.0f, 4.0f });
    set_values(weights, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });
    set_values(biases,  { -5.0f, -6.0f, -7.0f});

    auto conv = convolution::create({ engine::reference, output, {input, weights, biases}, { 1, {5, 5}, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ( 25.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ( 64.0f, output_ptr[1]);
    EXPECT_FLOAT_EQ(103.0f, output_ptr[2]);
}

TEST(convolution_f32_fw, basic_wsiz2x2x1x3_wstr2x2_in2x2x1x1_nopad) {
//  Filter : 2x2x1x3
//  Stride : 2x2
//  Input  : 2x2x1x1
//  Output : 1x1x3x1
//
//  Input:
//  -2.3 -0.1
//   3.1  1.9
//
//  Filter:
//  -1.1  1.5       0.1  0.2        2.0  -1.0
//   0.5 -0.5       0.4  0.7        2.5  -1.5
//
//  Bias:
//  0.1 -0.2 0.3
//
//  Output:
//   0.7
//   2.12
//   3.08

    auto input   = memory::allocate({ engine::reference, memory::format::yxfb_f32,{1 ,{2, 2}, 1}});
    auto output  = memory::allocate({ engine::reference, memory::format::yxfb_f32,{1 ,{1, 1}, 3}});
    auto weights = memory::allocate({ engine::reference, memory::format::oiyx_f32,{1 ,{2, 2},{3, 1}}});
    auto biases  = memory::allocate({ engine::reference, memory::format::   x_f32,{1 ,{{3 }}, 1}});

    set_values(input,   { -2.3f, -0.1f, 3.1f, 1.9f });
    set_values(weights, { -1.1f, 1.5f, 0.5f, -0.5f, 0.1f, 0.2f, 0.4f, 0.7f, 2.0f, -1.0f, 2.5f, -1.5f });
    set_values(biases,  { 0.1f, -0.2f, 0.3f });

    auto conv = convolution::create({ engine::reference, output, {input, weights, biases}, { 1, {2, 2}, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_TRUE(are_equal(3.08f, output_ptr[0]));
    EXPECT_TRUE(are_equal(2.12f, output_ptr[1]));
    EXPECT_TRUE(are_equal(0.7f , output_ptr[2]));
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
    auto input  = memory::allocate({engine::reference, memory::format::yxfb_f32, { 1, {2, 2}, 1}});
    auto output = memory::allocate({engine::reference, memory::format::yxfb_f32, { 1, {1, 1}, 1}});
    auto weights= memory::allocate({engine::reference, memory::format::oiyx_f32, { 1, {3, 3},{1, 1}}});
    auto biases = memory::allocate({engine::reference, memory::format::   x_f32, { 1, {{1}} , 1}});

    set_values(input  , {-0.5f, 1.0f, 0.5f, 2.0f});
    set_values(weights, {-2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f});
    set_values(biases , {2.0f});

    auto conv = convolution::create({engine::reference, output, {input, weights, biases}, {1, {2, 2}, 1}, padding::zero});
    execute({conv}).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(12.25f, output_ptr[0]);
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
//   rnd   2.0
    auto input  = memory::allocate({engine::cpu, memory::format::yxfb_f32, {1 ,{2, 2}, 1}});
    auto output = memory::allocate({engine::cpu, memory::format::yxfb_f32, {1 ,{2, 2}, 1}});
    auto weights= memory::allocate({engine::cpu, memory::format::oiyx_f32, {1 ,{3, 3},{1, 1}}});
    auto biases = memory::allocate({engine::cpu, memory::format::   x_f32, {1 ,{{1}} , 1}});

    set_values(input  , {-0.5f, 1.0f, 0.5f, 2.0f});
    set_values(weights, {-2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f});
    set_values(biases , {2.0f});

    auto conv = convolution::create({engine::reference,
                                     output,
                                     {0, {1, 1}, 0},
                                     {1, {1, 1}, 1},
                                     {input, weights, biases},
                                     {0, {-1, -1}, 0},
                                     {1, { 2,  2}, 1},
                                     padding::zero});
    execute({conv}).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(2.0f, output_ptr[3]);
}

TEST(convolution_f32_fw, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad_split2) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4x2
    //  Output : 2x2x2

    //  Input:
    //  f0: -0.5   1     0.5  2
    //       1.5  -0.5   0   -1
    //       0.5   0.5  -1    1
    //       0.5   2     1.5 -0.5
    //
    //  f1:  0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //       0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //
    //  Filter1:
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias1:
    //  2
    //
    //  Filter2:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias2:
    //  -1

    //  Output:
    //   8  3.65 0.5 -5.36
    //   6  3.65 9   -5.36

    auto input = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ 1,{ 4, 4 }, 2 } });
    auto output = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ 1,{ 2, 2 }, 2 } });
    auto weights1 = memory::allocate({ engine::reference, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases1 = memory::allocate({ engine::reference, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });
    auto weights2 = memory::allocate({ engine::reference, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases2 = memory::allocate({ engine::reference, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, {
        -0.5f,  0.5f,  1.0f,  1.5f,  0.5f,  2.3f,  2.0f, -0.4f,
         1.5f,  2.0f, -0.5f, -4.0f,  0.0f,  1.0f, -1.0f,  3.0f,
         0.5f,  0.5f,  0.5f,  1.5f, -1.0f,  2.3f,  1.0f, -0.4f,
         0.5f,  2.0f,  2.0f, -4.0f,  1.5f,  1.0f, -0.5f,  3.0f
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases1, { 2.0f });
    set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases2, { -1.0f });

    auto conv = convolution::create({
        engine::reference,
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2 
    });

    execute({ conv }).wait();

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_memory, 0));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_memory, 1));
    EXPECT_FLOAT_EQ(0.5f, get_value<float>(output_memory, 2));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_memory, 3));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_memory, 4));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_memory, 5));
    EXPECT_FLOAT_EQ(9.0f, get_value<float>(output_memory, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_memory, 7));
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
    auto bw_output    = memory::allocate({engine::reference, memory::format::yxfb_f32, {1 ,{3, 3}, 1}});
    auto bw_input     = memory::allocate({engine::reference, memory::format::yxfb_f32, {1 ,{2, 2}, 1}});
    auto fw_input     = memory::allocate({engine::reference, memory::format::yxfb_f32, {1 ,{3, 3}, 1}});
    auto weights      = memory::allocate({engine::reference, memory::format::yxfb_f32, {1 ,{2, 2}, 1}});
    auto weights_diff = memory::allocate({engine::reference, memory::format::yxfb_f32, {1 ,{2, 2}, 1}});
    auto biases       = memory::allocate({engine::reference, memory::format::x_f32,    {1 ,{{1}} , 1}});
    auto biases_diff  = memory::allocate({engine::reference, memory::format::x_f32,    {1 ,{{1}} , 1}});

    auto& bw_output_mem    = bw_output.as<const memory&>();
    auto& weights_diff_mem = weights_diff.as<const memory&>();
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

    execute({conv_bw}).wait();

    
    float bw_out_expected[] = { -2.0f, -0.5f, 7.0f, -5.5f, 5.0f, 17.0f, 1.5f, 6.5f, 6.0f };
    auto bw_output_ptr = bw_output_mem.pointer<float>();
    auto results_equal = std::equal(std::begin(bw_out_expected), std::end(bw_out_expected), std::begin(bw_output_ptr));
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong output gradient";

    results_equal = true;
    float weights_diff_expected[] = { -7.00f, 35.00f, 5.50f, 32.25f };
    auto weights_diff_ptr = weights_diff_mem.pointer<float>();
    results_equal = std::equal(std::begin(weights_diff_expected), std::end(weights_diff_expected), std::begin(weights_diff_ptr));
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong weights gradient";

    float biases_diff_expected[] = { 10.0f };
    auto biases_diff_ptr = biases_diff_mem.pointer<float>();
    results_equal = std::equal(std::begin(biases_diff_expected), std::end(biases_diff_expected), std::begin(biases_diff_ptr));
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong bias gradient";
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
//  -3
    auto bw_output    = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {2, 2}, 1}});
    auto bw_input     = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 1}});
    auto fw_input     = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {2, 2}, 1}});
    auto weights      = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {3, 3}, 1}});
    auto weights_diff = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {3, 3}, 1}});
    auto biases       = memory::allocate({engine::reference, memory::format::x_f32,    {1, {{1}} , 1}});
    auto biases_diff  = memory::allocate({engine::reference, memory::format::x_f32,    {1, {{1}} , 1}});

    auto& bw_output_mem    = bw_output.as<const memory&>();
    auto& weights_diff_mem = weights_diff.as<const memory&>();
    auto& biases_diff_mem  = biases_diff.as<const memory&>();

    set_values( fw_input, {-0.5f, 1.5f, 1.0f,-0.5f});
    set_values( bw_input, {2.0f});
    set_values( weights , {-2.0f, 3.5f, 1.0f, 0.5f, 1.5f, 2.0f, 1.0f, 2.0f, 3.0f});
    set_values( biases  , {-3.0f});

    auto conv_bw = convolution_backward::create({engine::reference,
                                                 std::vector<primitive>{bw_output, weights_diff, biases_diff},
                                                 {bw_input, fw_input, weights, biases},
                                                 {1, {1, 1}, 1},
                                                 padding::zero});
    execute({conv_bw}).wait();

    bool results_equal = true;
    float bw_output_expected[] = { -4.0f, 7.0f, 1.0f, 3.0f };
    auto bw_output_ptr = bw_output_mem.pointer<float>();
    results_equal = std::equal(std::begin(bw_output_expected), std::end(bw_output_expected), std::begin(bw_output_ptr));
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong output gradient";

    float weights_diff_expected[] = { 2.0f, 10.5f, 0.0f, 1.0f, -1.5f, 0.0f, 0.0f, 0.0f, 0.0f };
    auto weights_diff_ptr = weights_diff_mem.pointer<float>();
    results_equal = std::equal(std::begin(weights_diff_expected), std::end(weights_diff_expected), std::begin(weights_diff_ptr));
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong weights gradient";

    float biases_diff_expected[] = { 2.0f };
    auto biases_diff_ptr = biases_diff_mem.pointer<float>();
    results_equal = std::equal(std::begin(biases_diff_expected), std::end(biases_diff_expected), std::begin(biases_diff_ptr));
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong bias gradient";
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
//  -3
    using namespace neural;
    auto bw_output    = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {4, 4}, 1}});
    auto bw_input     = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {2, 2}, 1}});
    auto fw_input     = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {4, 4}, 1}});
    auto weights      = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {3, 3}, 1}});
    auto weights_diff = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {3, 3}, 1}});
    auto biases       = memory::allocate({engine::reference, memory::format::x_f32,    {1, {{1}} , 1}});
    auto biases_diff  = memory::allocate({engine::reference, memory::format::x_f32,    {1, {{1}} , 1}});

    auto& bw_output_mem    = bw_output.as<const memory&>();
    auto& bw_input_mem     = bw_input.as<const memory&>();
    auto& fw_input_mem     = fw_input.as<const memory&>();
    auto& weights_mem      = weights.as<const memory&>();
    auto& weights_diff_mem = weights_diff.as<const memory&>();
    auto& biases_mem       = biases.as<const memory&>();
    auto& biases_diff_mem  = biases_diff.as<const memory&>();

    fill(fw_input_mem, 1.0f);
    {
        auto ptr = fw_input_mem.pointer<float>();
        ptr[10] = -0.5f;
        ptr[11] = 1.5f;
        ptr[14] = 1.0f;
        ptr[15] = -0.5f;
    }
    fill(bw_input_mem, 1.0f);
    {
        auto ptr = bw_input_mem.pointer<float>();
        ptr[3] = 2.0f;
    }

    fill(weights_mem, 1.0f);
    {
        auto ptr = weights_mem.pointer<float>();
        ptr[4] = -2.0f;
        ptr[5] = 3.5f;
        ptr[7] = 0.5f;
        ptr[8] = 1.5f;
    }

    fill(biases_mem, -3.0f);

    auto conv_bw = convolution_backward::create({engine::reference,
                                                 std::vector<primitive>{bw_output, weights_diff, biases_diff},
                                                 {0, {1, 1}, 0},
                                                 {1, {1, 1}, 1},
                                                 {bw_input, fw_input, weights, biases},
                                                 {0, {1, 1}, 0},
                                                 {1, {1, 1}, 1},
                                                 padding::zero});
    execute({conv_bw}).wait();

    float bw_output_expected[] = { 0.0f, 0.0f,  0.0f, 0.0f,
                                   0.0f, 2.0f,  2.0f, 2.0f,
                                   0.0f, 2.0f, -4.0f, 7.0f,
                                   0.0f, 2.0f,  1.0f, 3.0f };
    auto bw_output_ptr = bw_output_mem.pointer<float>();
    auto results_equal = std::equal(std::begin(bw_output_expected), std::end(bw_output_expected), std::begin(bw_output_ptr));
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong output gradient";

    float weights_diff_expected[] = { 2.0f, 2.0f, 2.0f,
                                      2.0f, 2.0f, 10.5f,
                                      2.0f, 1.0f, -1.5f, };

    auto weights_diff_ptr = weights_diff_mem.pointer<float>();
    results_equal = std::equal(std::begin(weights_diff_expected), std::end(weights_diff_expected), std::begin(weights_diff_ptr));
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong weights gradient";

    float biases_diff_expected[] = { 2.0f };
    auto biases_diff_ptr = biases_diff_mem.pointer<float>();
    results_equal = std::equal(std::begin(biases_diff_expected), std::end(biases_diff_expected), std::begin(biases_diff_ptr));
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong bias gradient";
}

TEST(convolution_f32_fw, DISABLED_optimized_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {

    auto engine_resource = worker_cpu::create({std::thread::hardware_concurrency()});
    auto input           = memory::allocate({engine::cpu, memory::format::       byxf_f32,    {1, {4, 4}, 1}});
    auto output          = memory::allocate({engine::cpu, memory::format::       byxf_f32,   {1, {2, 2}, 16}});
    auto weights         = memory::allocate({engine::cpu, memory::format::os_yxi_sv16_f32, {{2, 2}, {16, 1}}});
    auto biases          = memory::allocate({engine::cpu, memory::format::          x_f32,   {1, {{16}} , 1}});

    auto& output_memory  = output.as<const memory&>();
    auto& input_memory   = input.as<const memory&>();
    auto& weights_memory = weights.as<const memory&>();
    auto& biases_memory  = biases.as<const memory&>();
    auto output_ptr = output_memory.pointer<float>();

    auto conv = convolution::create({engine::cpu, output, {input, weights, biases}, {1, {2, 2}, 1}, padding::zero});

    auto& kernel_sizes = weights_memory.argument.size;
    size_t num_input_kernel_elements = kernel_sizes.spatial[0] * kernel_sizes.spatial[1] * kernel_sizes.feature[1];
    size_t num_output_elements = output_memory.count();

    auto run_subtest = [&](float input_val, float weight_val, float bias_val)
    {
        fill<float>(input_memory, input_val);
        fill<float>(weights_memory, weight_val);
        fill<float>(biases_memory, bias_val);

        execute({conv}, {engine_resource}).wait();

        // This test was just a sum of uniform values in whole window.
        // So each output value should be: number of input elements of 3d kernel (times input and kernel values) + bias value.
        float expected_value = num_input_kernel_elements * input_val * weight_val + bias_val;
        for(size_t output_element = 0; output_element < num_output_elements; ++output_element)
            EXPECT_EQ(true, tests::are_equal(expected_value, output_ptr[output_element]));
    };

    run_subtest(1.0f, 1.0f, 1.0f);
    run_subtest(1.0f, 2.0f, 1.0f);
    run_subtest(1.0f, 1.0f, 2.0f);
    run_subtest(2.0f, 2.0f, 1.0f);
    run_subtest(1.0f, 2.0f, 2.0f);
    run_subtest(2.0f, 2.0f, 2.0f);
}

TEST(convolution_f32_fw, DISABLED_optimized_2slice_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {

    // This implementation will use two jobs, each for one slice, so make sure it will test MT path, no matter what underlying HW we have.
    auto engine_resource = worker_cpu::create({2});

    auto input           = memory::allocate({engine::cpu, memory::format::       byxf_f32,    {1, {4, 4}, 1}});
    auto output          = memory::allocate({engine::cpu, memory::format::       byxf_f32,   {1, {2, 2}, 32}});
    auto weights         = memory::allocate({engine::cpu, memory::format::os_yxi_sv16_f32, {{2, 2}, {32, 1}}});
    auto biases          = memory::allocate({engine::cpu, memory::format::          x_f32,   {1, {{32}} , 1}});

    auto& output_memory  = output.as<const memory&>();
    auto& input_memory   = input.as<const memory&>();
    auto& weights_memory = weights.as<const memory&>();
    auto& biases_memory  = biases.as<const memory&>();

    auto conv = convolution::create({engine::cpu, output, {input, weights, biases}, {1, {2, 2}, 1}, padding::zero});

    auto& kernel_sizes = weights_memory.argument.size;
    auto& output_sizes = output_memory.argument.size;
    size_t num_input_kernel_elements = kernel_sizes.spatial[0] * kernel_sizes.spatial[1] * kernel_sizes.feature[1];

    auto run_subtest = [&](float input_val, float weight_val_slice0, float weight_val_slice1, float bias_val_slice0, float bias_val_slice1)
    {
        fill<float>(input_memory, input_val);

        // Weights and biases are grouped by slices, so we can easily initialize them with different values.
        {// restrict memory::ptr lifetime
            auto weights_ptr = weights_memory.pointer<float>();
            auto weights_count_half = weights_memory.count() / 2;
            std::fill(std::begin(weights_ptr), std::begin(weights_ptr) + weights_count_half, weight_val_slice0);
            std::fill(std::begin(weights_ptr) + weights_count_half, std::end(weights_ptr), weight_val_slice1);
        }

        {// restrict memory::ptr lifetime
            auto biases_ptr = biases_memory.pointer<float>();
            auto biases_count_half = biases_memory.count() / 2;
            std::fill(std::begin(biases_ptr), std::begin(biases_ptr) + biases_count_half, bias_val_slice0);
            std::fill(std::begin(biases_ptr) + biases_count_half, std::end(biases_ptr), bias_val_slice1);
        }

        execute({conv}, {engine_resource}).wait();

        // This test was just a sum of uniform values in whole window.
        // So each output value should be: number of input elements of 3d kernel (times input and kernel values) + bias value.
        float expected_value_slice0 = num_input_kernel_elements * input_val * weight_val_slice0 + bias_val_slice0;
        float expected_value_slice1 = num_input_kernel_elements * input_val * weight_val_slice1 + bias_val_slice1;

        auto output_ptr = output_memory.pointer<float>();
        for(uint32_t fmap = 0; fmap < output_sizes.feature[0]; ++fmap)
            for(uint32_t col = 0; col < output_sizes.spatial[0]; ++col)
                for(uint32_t row = 0; row < output_sizes.spatial[1]; ++row)
                {
                    uint32_t output_element = fmap + col * output_sizes.feature[0] + row * output_sizes.feature[0] * output_sizes.spatial[0];
                    EXPECT_EQ(true, tests::are_equal((fmap < 16) ? expected_value_slice0 : expected_value_slice1, output_ptr[output_element]));
                }
        };

    run_subtest(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    run_subtest(2.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    run_subtest(1.0f, 2.0f, 1.0f, 1.0f, 1.0f);
    run_subtest(1.0f, 1.0f, 2.0f, 1.0f, 1.0f);
    run_subtest(1.0f, 1.0f, 1.0f, 2.0f, 1.0f);
    run_subtest(1.0f, 1.0f, 1.0f, 2.0f, 2.0f);

    run_subtest(2.0f, 1.0f, 1.0f, 1.0f, 3.0f);
    run_subtest(1.0f, 2.0f, 1.0f, 3.0f, 1.0f);
    run_subtest(1.0f, 1.0f, 5.0f, 1.0f, 1.0f);
    run_subtest(1.0f, 3.0f, 1.0f, 2.0f, 1.0f);
    run_subtest(3.0f, 1.0f, 1.0f, 2.0f, 2.0f);
}

TEST(convolution_f32_fw, DISABLED_naive_comparison_optimized_2slice_wsiz3x3_wstr2x3_in21x12x3x2_nopad) {

    // This implementation will use two jobs, each for one slice, so make sure it will test MT path, no matter what underlying HW we have.
    auto engine_resource = worker_cpu::create({2});

    // Optimized data.
    auto input   = memory::allocate({engine::cpu, memory::format::       byxf_f32,  {2, {11, 12}, 3}}); auto& input_memory   = input.as<const memory&>();
    auto output  = memory::allocate({engine::cpu, memory::format::       byxf_f32,   {2, {5, 4}, 32}});
    auto weights = memory::allocate({engine::cpu, memory::format::os_yxi_sv16_f32, {{3, 2}, {32, 3}}}); auto& weights_memory = weights.as<const memory&>();
    auto biases  = memory::allocate({engine::cpu, memory::format::          x_f32,   {1, {{32}} , 1}}); auto& biases_memory  = biases.as<const memory&>();

    // Reference data.
    auto ref_input   = memory::allocate({engine::cpu, memory::format::yxfb_f32,  {2, {11, 12}, 3}});
    auto ref_output  = memory::allocate({engine::cpu, memory::format::yxfb_f32,   {2, {5, 4}, 32}}); auto& ref_output_memory  = ref_output.as<const memory&>();
    auto ref_weights = memory::allocate({engine::cpu, memory::format::oiyx_f32, {{3, 2}, {32, 3}}});
    auto ref_biases  = memory::allocate({engine::cpu, memory::format::   x_f32,   {1, {{32}} , 1}});

    // Temporary data for optimized results in reference space.
    auto temp_output  = memory::allocate({engine::cpu, memory::format::yxfb_f32, {2, {5, 4}, 32}}); auto& temp_output_memory = temp_output.as<const memory&>();

    // Reordering primitives.
    auto reorder_input_to_ref   = reorder::create({engine::reference, input, ref_input});
    auto reorder_weights_to_ref = reorder::create({engine::reference, weights, ref_weights});
    auto reorder_biases_to_ref  = reorder::create({engine::reference, biases, ref_biases});
    auto reorder_output_to_tmp_ref  = reorder::create({engine::reference, output, temp_output});

    // Main convolutions.
    auto opt_conv = convolution::create({engine::cpu, output, {input, weights, biases}, {1, {2, 3}, 1}, padding::zero});
    auto ref_conv = convolution::create({engine::reference, ref_output, {ref_input, ref_weights, ref_biases}, {1, {2, 3}, 1}, padding::zero});

    // Initialize data.
    fill_rng<float>(input_memory, 10, -5.0f, 5.0f);
    fill_rng<float>(weights_memory, 11, -5.0f, 5.0f);
    fill_rng<float>(biases_memory, 12, -5.0f, 5.0f);

    execute(
    {
        opt_conv,
        reorder_output_to_tmp_ref,
        reorder_input_to_ref,
        reorder_weights_to_ref,
        reorder_biases_to_ref,
        ref_conv,
    }, {engine_resource}).wait();

    {
        auto ref_out_ptr = ref_output_memory.pointer<float>();
        auto temp_out_ptr = temp_output_memory.pointer<float>();
        for(size_t output_element = 0; output_element < temp_output_memory.count(); ++output_element)
            EXPECT_EQ(true, tests::are_equal(ref_out_ptr[output_element],
                                             temp_out_ptr[output_element],
                                             0.0005f));
    }
}

TEST(convolution_f32_fw, DISABLED_optimized_generic_vs_for_loop_implementation) {
    const uint32_t input_width         = 20;   // data = input|output
    const uint32_t input_height        = 20;   // data = input|output
    const uint32_t input_feature_maps  = 8;
    const uint32_t input_view_start_x  = 0;
    const uint32_t input_view_start_y  = 0;
    const uint32_t stride_width        = 1;
    const uint32_t stride_height       = 1;
    const uint32_t output_width        = (input_width +stride_width -1)/stride_width;
    const uint32_t output_height       = (input_height+stride_height-1)/stride_height;
    const uint32_t output_feature_maps = 8;
    const uint32_t output_view_x       = 0;
    const uint32_t output_view_y       = 0;
    const uint32_t output_view_width   = (input_width +stride_width -1)/stride_width;
    const uint32_t output_view_height  = (input_height+stride_height-1)/stride_height;
    const uint32_t filter_size         = 5;    // filter size is the same for both axes
    const uint32_t output_features_per_iteration = 4;

    // allocate memory buffers
    const uint64_t batch_size = 24;
    const auto align_size = 64;
    const auto align_size_in_float = align_size/sizeof(float);
    const auto output_buffer_size = output_height*output_width*output_feature_maps*batch_size;
    const auto filter_buffer_size = filter_size*filter_size*output_feature_maps*input_feature_maps;
    const auto  input_buffer_size =  input_height* input_width* input_feature_maps*batch_size;
    const auto   bias_buffer_size = output_feature_maps;

    std::unique_ptr<float> output_container = std::move(std::unique_ptr<float>(new float[output_buffer_size+align_size_in_float]));
    std::unique_ptr<float> filter_container = std::move(std::unique_ptr<float>(new float[filter_buffer_size+align_size_in_float]));
    std::unique_ptr<float>  input_container = std::move(std::unique_ptr<float>(new float[ input_buffer_size+align_size_in_float]));
    std::unique_ptr<float>   bias_container = std::move(std::unique_ptr<float>(new float[  bias_buffer_size+align_size_in_float]));

    const auto output = output_container.get();
    const auto filter = filter_container.get();
    const auto  input =  input_container.get();
    const auto   bias =   bias_container.get();

    auto input_p  = memory::describe({neural::engine::reference, memory::format::byxf_b24_f32, { 24 , {input_width , input_height }, input_feature_maps}});
    auto output_p = memory::describe({neural::engine::reference, memory::format::byxf_b24_f32, { 24 , {output_width, output_height}, output_feature_maps}});
    auto weights_p= memory::describe({neural::engine::reference, memory::format::yxoi_o4_f32 , { 1  , {filter_size   , filter_size }, {output_feature_maps, input_feature_maps}}});
    auto biases_p = memory::describe({neural::engine::reference, memory::format::x_f32       , { 1  , {{output_feature_maps}}  , 1 }});

    // initialized inputs & filter with pseudorandom values
    std::mt19937 engine(0xdeadf00d);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    auto lambda = [&]{return distribution(engine);};
    std::fill    (output, output+output_buffer_size, 0.0f);
    std::generate( input,  input+ input_buffer_size, lambda);
    std::generate( bias,  bias+ bias_buffer_size, lambda);
    std::generate(filter, filter+filter_buffer_size, lambda);

    auto engine_resource = worker_cpu::create({2});

    //set pointers
    execute(
    {output_p(output), input_p(input), weights_p(filter), biases_p(bias)}
    , {engine_resource}).wait();

    auto conv   = convolution::create( {neural::engine::cpu,
                                        output_p,
                                        {input_p, weights_p, biases_p},
                                        {1, {stride_height, stride_width}, 1},
                                        padding::zero}
                                        );

    execute(
        {conv}
    , {engine_resource}).wait();

    const int64_t filter_radius = (filter_size-1)/2;
    const auto output_feature_blocks    = output_feature_maps/output_features_per_iteration;
    for(uint64_t b=0; b<batch_size; ++b) {
        for(uint64_t y=0; y<output_view_height; ++y) {
            for(uint64_t x=0; x<output_view_width; ++x)
                for(uint64_t fo=0; fo<output_feature_maps; ++fo) {
                    int64_t foi = fo%output_features_per_iteration;
                    int64_t fob = fo/output_features_per_iteration;
                    float valid = bias[fo];
                    for(uint64_t yk=0; yk<filter_size; ++yk) {
                        int64_t ys = static_cast<int64_t>(input_view_start_y) + y*stride_height+yk-filter_radius;
                        if(ys<0 || static_cast<uint64_t>(ys)>=input_height) continue;
                        for(uint64_t xk=0; xk<filter_size; ++xk) {
                            int64_t xs = static_cast<int64_t>(input_view_start_x) + x*stride_height+xk-filter_radius;
                            if(xs<0 || static_cast<uint64_t>(xs)>=input_width) continue;
                            for(uint64_t fi=0; fi<input_feature_maps; ++fi) {
                                float value  = input[b + batch_size*(fi + input_feature_maps*(xs + input_width*ys))];
                                float weight = filter[foi + output_features_per_iteration*(fi + input_feature_maps*(fob + output_feature_blocks*(xk + filter_size*yk)))];
                                valid  = static_cast<float>(double(value)*weight + valid);
                            }
#if 0
                            std::cout << "[yo=" << yo << ",xo=" << xo << ",fo=" << fo << ",yk=" << yk << ",xk=" << xk << "]: " << valid;
                            std::cout << std::endl;
#endif
                        }
                    }
                    if(valid<0) valid = 0;
                    auto yo = y+output_view_y;
                    auto xo = x+output_view_x;
                    auto tested = output[b + batch_size*(fo + output_feature_maps*(xo + output_width*yo))];
                    EXPECT_EQ(true, tests::are_equal(valid, tested)) << "at [b,x,y,f] = [" << b << "," << xo << "," << yo << "," << fo << "]\n" << std::endl;
                }
        }
    }
}

TEST(convolution_f32_fw, DISABLED_optimized_generic_vs_ref_implementation) {
    const uint32_t input_width         = 6;   // data = input|output
    const uint32_t input_height        = 6;   // data = input|output
    const uint32_t input_feature_maps  = 8;
    const uint32_t stride_width        = 2;
    const uint32_t stride_height       = 2;
    const uint32_t output_width        = (input_width +stride_width -1)/stride_width;
    const uint32_t output_height       = (input_height+stride_height-1)/stride_height;
    const uint32_t output_feature_maps = 4;
    const uint32_t filter_size         = 2;    // filter size is the same for both axes

    // allocate memory buffers
    const uint64_t batch_size = 24;
    const auto output_buffer_size = output_height*output_width*output_feature_maps*batch_size;
    const auto filter_buffer_size = filter_size*filter_size*output_feature_maps*input_feature_maps;
    const auto  input_buffer_size =  input_height* input_width* input_feature_maps*batch_size;
    const auto   bias_buffer_size = output_feature_maps;

    std::unique_ptr<float> output_container = std::move(std::unique_ptr<float>(new float[output_buffer_size]));
    std::unique_ptr<float> filter_container = std::move(std::unique_ptr<float>(new float[filter_buffer_size]));
    std::unique_ptr<float>  input_container = std::move(std::unique_ptr<float>(new float[ input_buffer_size]));
    std::unique_ptr<float>   bias_container = std::move(std::unique_ptr<float>(new float[  bias_buffer_size]));

    const auto output =output_container.get();
    const auto filter =filter_container.get();
    const auto  input = input_container.get();
    const auto   bias =  bias_container.get();

    // Memory descriptors for optimized convolution
    auto output_optimized = memory::describe({neural::engine::reference, memory::format::byxf_b24_f32, { 24 , {output_width, output_height}, output_feature_maps}});
    auto input_optimized  = memory::describe({neural::engine::reference, memory::format::byxf_b24_f32, { 24 , {input_width , input_height }, input_feature_maps}});
    auto weights_optimized= memory::describe({neural::engine::reference, memory::format::yxoi_o4_f32 , { 1  , {filter_size   , filter_size}, {output_feature_maps, input_feature_maps}}});
    auto biases           = memory::describe({neural::engine::reference, memory::format::x_f32       , { 1  , {{output_feature_maps}}      , 1 }});
    auto output_optimized_in_ref_format = memory::allocate({neural::engine::reference, memory::format::yxfb_f32, { 24 , {output_width, output_height}, output_feature_maps}});

    // Set pointers
    execute({output_optimized(output), input_optimized(input), weights_optimized(filter), biases(bias)});

    // Fill primitives with random data
    fill<float>(weights_optimized, 1.0f);
    fill<float>(output_optimized , 1.0f);
    fill<float>(input_optimized  , 1.0f);
    fill<float>(biases           , 1.0f);

    auto conv_optimized   = convolution::create( {neural::engine::cpu,
                                                 output_optimized,
                                                 {input_optimized, weights_optimized, biases},
                                                 {1, {stride_height, stride_width}, 1},
                                                 padding::zero}
                                                );

     // Memory for reference convolution
    auto input_ref  = memory::allocate({neural::engine::reference, memory::format::yxfb_f32, { 24 , {input_width , input_height }, input_feature_maps}});
    auto output_ref = memory::allocate({neural::engine::reference, memory::format::yxfb_f32, { 24 , {output_width, output_height}, output_feature_maps}});
    auto weights_ref= memory::allocate({neural::engine::reference, memory::format::oiyx_f32, { 1  , {filter_size , filter_size  }, {output_feature_maps, input_feature_maps}}});

    // the same bias primitive is used in both implementations
    auto conv_ref   = convolution::create( {neural::engine::reference,
                                            output_ref,
                                            {input_ref, weights_ref, biases},
                                            {1, {stride_height, stride_width}, 1},
                                            padding::zero}
                                           );

    // Reordering primitives.
    auto reorder_input_optimized_to_ref   = reorder::create({engine::reference, input_optimized, input_ref});
    auto reorder_weights_optimized_to_ref = reorder::create({engine::reference, weights_optimized, weights_ref});

    auto reorder_output_optimized_ref_format = reorder::create({engine::reference, output_optimized, output_optimized_in_ref_format});

    auto engine_resource = worker_cpu::create({4});
     execute(
     { reorder_input_optimized_to_ref, reorder_weights_optimized_to_ref // copy input/weigts
     , conv_optimized, conv_ref                                         // run both convolutions
     , reorder_output_optimized_ref_format }                            // copy optimized output to another buffer in reference output's format, so we can compare them
     , {engine_resource}).wait();

     auto output_mem_optimized = output_optimized.as<const memory&>().pointer<float>();
     auto output_mem_ref       = output_ref.as<const memory&>().pointer<float>();
     for(size_t i = 0; i < output_buffer_size; ++i)
       EXPECT_EQ(true, tests::are_equal(output_mem_ref[i],
                                        output_mem_optimized[i]))
               << "at index " << i;
}