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

using namespace neural;
using namespace tests;

namespace{
template<typename T> T *align(T *pointer, size_t align) {
    return reinterpret_cast<T *>((reinterpret_cast<uintptr_t>(pointer)+align-1)/align*align);
};
}

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

    auto input  = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {4, 4}, 1}});
    auto output = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {2, 2}, 1}});
    auto weights= memory::allocate({engine::reference, memory::format::oiyx_f32, {1, {2, 2},{1, 1}}});
    auto biases = memory::allocate({engine::reference, memory::format::   x_f32, {1, {{1}} , 1}});

    set_values(input  , {-0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f});
    set_values(weights, {-2.0f, 0.5f, 3.5f, 1.5f});
    set_values(biases , {2.0f});

    auto conv = convolution::create({engine::reference, output, {input, weights, biases}, {1, {2, 2}, 1}, padding::zero});

    execute({conv}).sync();

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_memory, 0));
    EXPECT_FLOAT_EQ(0.5f, get_value<float>(output_memory, 1));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_memory, 2));
    EXPECT_FLOAT_EQ(9.0f, get_value<float>(output_memory, 3));
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
    auto input   = memory::allocate({ engine::reference, memory::format::yxfb_f32,{2, {2, 2}, 1}});
    auto output  = memory::allocate({ engine::reference, memory::format::yxfb_f32,{2, {1, 1}, 1}});
    auto weights = memory::allocate({ engine::reference, memory::format::oiyx_f32,{1, {2, 2},{1, 1}}});
    auto biases  = memory::allocate({ engine::reference, memory::format::   x_f32,{1, {{1}} , 1}});

    set_values(input,  { 0.5f, 2.3f, 1.5f, -0.4f, 2.0f, 1.0f, -4.0f, 3.0f });
    set_values(weights,{ -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases, { -1.0f });

    auto conv = convolution::create({ engine::reference, output, {input, weights, biases}, { 1, {2, 2}, 1 }, padding::zero });

    execute({ conv }).sync();

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_memory, 0));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_memory, 1));
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

    execute({ conv }).sync();

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(5.1f, get_value<float>(output_memory, 0));
    EXPECT_FLOAT_EQ(-5.2f, get_value<float>(output_memory, 1));
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

    execute({ conv }).sync();

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ( 25.0f, get_value<float>(output_memory, 0));
    EXPECT_FLOAT_EQ( 64.0f, get_value<float>(output_memory, 1));
    EXPECT_FLOAT_EQ(103.0f, get_value<float>(output_memory, 2));
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

    execute({ conv }).sync();

    auto& output_memory = output.as<const memory&>();
    EXPECT_TRUE(are_equal(3.08f, get_value<float>(output_memory, 0)));
    EXPECT_TRUE(are_equal(2.12f, get_value<float>(output_memory, 1)));
    EXPECT_TRUE(are_equal(0.7f , get_value<float>(output_memory, 2)));
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
    execute({conv}).sync();

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(12.25f, get_value<float>(output_memory, 0));
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
    execute({conv}).sync();

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(2.0f, get_value<float>(output_memory, 3));
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

    execute({conv_bw}).sync();

    bool results_equal = true;
    results_equal &= -2.0f == get_value<float>(bw_output_mem, 0);
    results_equal &= -0.5f == get_value<float>(bw_output_mem, 1);
    results_equal &=  7.0f == get_value<float>(bw_output_mem, 2);
    results_equal &= -5.5f == get_value<float>(bw_output_mem, 3);
    results_equal &=  5.0f == get_value<float>(bw_output_mem, 4);
    results_equal &= 17.0f == get_value<float>(bw_output_mem, 5);
    results_equal &=  1.5f == get_value<float>(bw_output_mem, 6);
    results_equal &=  6.5f == get_value<float>(bw_output_mem, 7);
    results_equal &=  6.0f == get_value<float>(bw_output_mem, 8);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong output gradient";

    results_equal = true;
    results_equal &= -7.00f == get_value<float>(weights_diff_mem, 0);
    results_equal &= 35.00f == get_value<float>(weights_diff_mem, 1);
    results_equal &=  5.50f == get_value<float>(weights_diff_mem, 2);
    results_equal &= 32.25f == get_value<float>(weights_diff_mem, 3);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong weights gradient";

    results_equal = true;
    results_equal &= 10.0f == get_value<float>(biases_diff_mem, 0);
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
    execute({conv_bw}).sync();

    bool results_equal = true;
    results_equal &= -4.0f == get_value<float>(bw_output_mem, 0);
    results_equal &=  7.0f == get_value<float>(bw_output_mem, 1);
    results_equal &=  1.0f == get_value<float>(bw_output_mem, 2);
    results_equal &=  3.0f == get_value<float>(bw_output_mem, 3);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong output gradient";

    results_equal = true;
    results_equal &=  2.0f == get_value<float>(weights_diff_mem, 0);
    results_equal &= 10.5f == get_value<float>(weights_diff_mem, 1);
    results_equal &=  0.0f == get_value<float>(weights_diff_mem, 2);
    results_equal &=  1.0f == get_value<float>(weights_diff_mem, 3);
    results_equal &= -1.5f == get_value<float>(weights_diff_mem, 4);
    results_equal &=  0.0f == get_value<float>(weights_diff_mem, 5);
    results_equal &=  0.0f == get_value<float>(weights_diff_mem, 6);
    results_equal &=  0.0f == get_value<float>(weights_diff_mem, 7);
    results_equal &=  0.0f == get_value<float>(weights_diff_mem, 8);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong weights gradient";

    results_equal = true;
    results_equal &= 2.0f == get_value<float>(biases_diff_mem, 0);
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
    static_cast<float*>(fw_input_mem.pointer)[10] = -0.5f;
    static_cast<float*>(fw_input_mem.pointer)[11] =  1.5f;
    static_cast<float*>(fw_input_mem.pointer)[14] =  1.0f;
    static_cast<float*>(fw_input_mem.pointer)[15] = -0.5f;

    fill(bw_input_mem, 1.0f);
    static_cast<float*>(bw_input_mem.pointer)[3] = 2.0f;

    fill(weights_mem, 1.0f);
    static_cast<float*>(weights_mem.pointer)[4] = -2.0f;
    static_cast<float*>(weights_mem.pointer)[5] =  3.5f;
    static_cast<float*>(weights_mem.pointer)[7] =  0.5f;
    static_cast<float*>(weights_mem.pointer)[8] =  1.5f;

    fill(biases_mem, -3.0f);

    auto conv_bw = convolution_backward::create({engine::reference,
                                                 std::vector<primitive>{bw_output, weights_diff, biases_diff},
                                                 {0, {1, 1}, 0},
                                                 {1, {1, 1}, 1},
                                                 {bw_input, fw_input, weights, biases},
                                                 {0, {1, 1}, 0},
                                                 {1, {1, 1}, 1},
                                                 padding::zero});
    execute({conv_bw}).sync();

    bool results_equal = true;
    results_equal &=  0.0f == get_value<float>(bw_output_mem, 0);
    results_equal &=  0.0f == get_value<float>(bw_output_mem, 1);
    results_equal &=  0.0f == get_value<float>(bw_output_mem, 2);
    results_equal &=  0.0f == get_value<float>(bw_output_mem, 3);
    results_equal &=  0.0f == get_value<float>(bw_output_mem, 4);
    results_equal &=  2.0f == get_value<float>(bw_output_mem, 5);
    results_equal &=  2.0f == get_value<float>(bw_output_mem, 6);
    results_equal &=  2.0f == get_value<float>(bw_output_mem, 7);
    results_equal &=  0.0f == get_value<float>(bw_output_mem, 8);
    results_equal &=  2.0f == get_value<float>(bw_output_mem, 9);
    results_equal &= -4.0f == get_value<float>(bw_output_mem, 10);
    results_equal &=  7.0f == get_value<float>(bw_output_mem, 11);
    results_equal &=  0.0f == get_value<float>(bw_output_mem, 12);
    results_equal &=  2.0f == get_value<float>(bw_output_mem, 13);
    results_equal &=  1.0f == get_value<float>(bw_output_mem, 14);
    results_equal &=  3.0f == get_value<float>(bw_output_mem, 15);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong output gradient";

    results_equal = true;
    results_equal &=  2.0f == get_value<float>(weights_diff_mem, 0);
    results_equal &=  2.0f == get_value<float>(weights_diff_mem, 1);
    results_equal &=  2.0f == get_value<float>(weights_diff_mem, 2);
    results_equal &=  2.0f == get_value<float>(weights_diff_mem, 3);
    results_equal &=  2.0f == get_value<float>(weights_diff_mem, 4);
    results_equal &= 10.5f == get_value<float>(weights_diff_mem, 5);
    results_equal &=  2.0f == get_value<float>(weights_diff_mem, 6);
    results_equal &=  1.0f == get_value<float>(weights_diff_mem, 7);
    results_equal &= -1.5f == get_value<float>(weights_diff_mem, 8);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong weights gradient";

    results_equal = true;
    results_equal &= 2.0f == get_value<float>(biases_diff_mem, 0);
    EXPECT_TRUE(results_equal) << "ERROR MESSAGE: wrong bias gradient";
}

TEST(convolution_f32_fw, optimized_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {

    auto engine_resource = worker_cpu::create({std::thread::hardware_concurrency()});
    auto input           = memory::allocate({engine::cpu, memory::format::       byxf_f32,    {1, {4, 4}, 1}});
    auto output          = memory::allocate({engine::cpu, memory::format::       byxf_f32,   {1, {2, 2}, 16}});
    auto weights         = memory::allocate({engine::cpu, memory::format::os_yxi_sv16_f32, {{2, 2}, {16, 1}}});
    auto biases          = memory::allocate({engine::cpu, memory::format::          x_f32,   {1, {{16}} , 1}});

    auto& output_memory  = output.as<const memory&>();
    auto& input_memory   = input.as<const memory&>();
    auto& weights_memory = weights.as<const memory&>();
    auto& biases_memory  = biases.as<const memory&>();

    auto conv = convolution::create({engine::cpu, output, {input, weights, biases}, {1, {2, 2}, 1}, padding::zero});

    auto& kernel_sizes = weights_memory.argument.size;
    size_t num_input_kernel_elements = kernel_sizes.spatial[0] * kernel_sizes.spatial[1] * kernel_sizes.feature[1];
    size_t num_output_elements = output_memory.count();

    auto run_subtest = [&](float input_val, float weight_val, float bias_val)
    {
        fill<float>(input_memory, input_val);
        fill<float>(weights_memory, weight_val);
        fill<float>(biases_memory, bias_val);

        execute({conv}, engine_resource).sync();

        // This test was just a sum of uniform values in whole window. 
        // So each output value should be: number of input elements of 3d kernel (times input and kernel values) + bias value.
        float expected_value = num_input_kernel_elements * input_val * weight_val + bias_val;
        for(size_t output_element = 0; output_element < num_output_elements; ++output_element)
            EXPECT_EQ(true, tests::are_equal(expected_value, get_value<float>(output_memory, output_element)));
    };

    run_subtest(1.0f, 1.0f, 1.0f);
    run_subtest(1.0f, 2.0f, 1.0f);
    run_subtest(1.0f, 1.0f, 2.0f);
    run_subtest(2.0f, 2.0f, 1.0f);
    run_subtest(1.0f, 2.0f, 2.0f);
    run_subtest(2.0f, 2.0f, 2.0f);
}

TEST(convolution_f32_fw, optimized_2slice_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {

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
        for(size_t weight_element = 0; weight_element < weights_memory.count(); ++weight_element)
            set_value<float>(weights_memory, weight_element, (weight_element < weights_memory.count()/2) ? weight_val_slice0 : weight_val_slice1);

        for(size_t bias_element = 0; bias_element < biases_memory.count(); ++bias_element)
            set_value<float>(biases_memory, bias_element, (bias_element < biases_memory.count()/2) ? bias_val_slice0 : bias_val_slice1);

        execute({conv}, engine_resource).sync();

        // This test was just a sum of uniform values in whole window. 
        // So each output value should be: number of input elements of 3d kernel (times input and kernel values) + bias value.
        float expected_value_slice0 = num_input_kernel_elements * input_val * weight_val_slice0 + bias_val_slice0;
        float expected_value_slice1 = num_input_kernel_elements * input_val * weight_val_slice1 + bias_val_slice1;

        for(uint32_t fmap = 0; fmap < output_sizes.feature[0]; ++fmap)
            for(uint32_t col = 0; col < output_sizes.spatial[0]; ++col)
                for(uint32_t row = 0; row < output_sizes.spatial[1]; ++row)
                {
                    uint32_t output_element = fmap + col * output_sizes.feature[0] + row * output_sizes.feature[0] * output_sizes.spatial[0];
                    EXPECT_EQ(true, tests::are_equal((fmap < 16) ? expected_value_slice0 : expected_value_slice1, get_value<float>(output_memory, output_element)));
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

TEST(convolution_f32_fw, naive_comparison_optimized_2slice_wsiz3x3_wstr2x3_in21x12x3x2_nopad) {

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
    }, engine_resource).sync();

    for(size_t output_element = 0; output_element < temp_output_memory.count(); ++output_element)
        EXPECT_EQ(true, tests::are_equal(get_value<float>(ref_output_memory, output_element), get_value<float>(temp_output_memory, output_element), 0.0005f)); 
}

TEST(convolution_f32_fw, optimized_generic_vs_for_loop_implementation) {
    const uint32_t input_width         = 40;   // data = input|output
    const uint32_t input_height        = 40;   // data = input|output
    const uint32_t input_feature_maps  = 8;
    const uint32_t input_view_start_x  = 0;
    const uint32_t input_view_start_y  = 0;
    const uint32_t stride_width        = 1;
    const uint32_t stride_height       = 1;
    const uint32_t output_width        = (input_width +stride_width -1)/stride_width;
    const uint32_t output_height       = (input_height+stride_height-1)/stride_height;
    const uint32_t output_feature_maps = 4;
    const uint32_t output_view_x       = 0;
    const uint32_t output_view_y       = 0;
    const uint32_t output_view_width   = (input_width +stride_width -1)/stride_width;
    const uint32_t output_view_height  = (input_height+stride_height-1)/stride_height;
    const uint32_t filter_size         = 5;    // filter size is the same for both axes
    const uint32_t output_features_per_iteration = 4;

    try {
    // allocate memory buffers
        const uint64_t batch_size = 24;
        const auto align_size = 64;
        const auto align_size_in_float = align_size/sizeof(float);
        const auto output_buffer_size = output_height*output_width*output_feature_maps*batch_size;
        const auto  input_buffer_size =  input_height* input_width* input_feature_maps*batch_size;
        const auto filter_buffer_size = filter_size*filter_size*output_feature_maps*input_feature_maps;

        std::unique_ptr<float> output_container = std::move(std::unique_ptr<float>(new float[output_buffer_size+align_size_in_float]));
        std::unique_ptr<float>  input_container = std::move(std::unique_ptr<float>(new float[ input_buffer_size+align_size_in_float]));
        std::unique_ptr<float> filter_container = std::move(std::unique_ptr<float>(new float[filter_buffer_size+align_size_in_float]));

        const auto output = align(output_container.get(), align_size);
        const auto  input = align( input_container.get(), align_size);
        const auto filter = align(filter_container.get(), align_size);

        auto input_p  = memory::describe({neural::engine::reference, memory::format::tmp_format, { 24 , {input_height  , input_width }, input_feature_maps}});
        auto output_p = memory::describe({neural::engine::reference, memory::format::tmp_format, { 24 , {output_height , output_width}, output_feature_maps}});
        auto weights_p= memory::describe({neural::engine::reference, memory::format::oiyx_f32, { 1  , {filter_size   , filter_size }, {output_feature_maps, input_feature_maps}}});
        auto biases_p = memory::allocate({neural::engine::reference, memory::format::   x_f32, { 1  , {{output_feature_maps}}  , 1 }});
        fill(biases_p, 0.0f);

        // initialized inputs & filter with pseudorandom values
        std::mt19937 engine(0xdeadf00d);
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        auto lambda = [&]{return distribution(engine);};
        std::fill    (output, output+output_buffer_size, 0.0f);
        std::generate( input,  input+ input_buffer_size, lambda);
        std::generate(filter, filter+filter_buffer_size, lambda);

        //set pointers
        execute(
            {output_p(output), input_p(input), weights_p(filter)}
        ).sync();

        auto conv   = convolution::create( {neural::engine::cpu,
                                            output_p,
                                            {input_p, weights_p, biases_p},
                                            {1, {stride_height, stride_width}, 1},
                                            padding::zero}
                                          );

        auto engine_resource = worker_cpu::create({2});
        execute(
            {conv}
           , engine_resource
        ).sync();

        const int64_t filter_radius = (filter_size-1)/2;
        const auto output_feature_blocks    = output_feature_maps/output_features_per_iteration;
        for(uint64_t b=0; b<batch_size; ++b) {
            for(uint64_t y=0; y<output_view_height; ++y) {
                for(uint64_t x=0; x<output_view_width; ++x)
                    for(uint64_t fo=0; fo<output_feature_maps; ++fo) {
                        int64_t foi = fo%output_features_per_iteration;
                        int64_t fob = fo/output_features_per_iteration;
                        float valid = 0.0f;
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
                        EXPECT_EQ(true, tests::are_equal(valid, tested)) << "at [b,x,y,f] = [" << b << "," << xo << "," << yo << "," << fo << "]\n"
                                                                            << "\tvalid  = " << valid << "\n"
                                                                            << "\ttested = " << tested << std::endl;
                    }
            }
        }
    } catch (std::exception &e) {
        std::cerr << e.what();
    } catch (...) {
        std::cerr << "Unknown exceptions.";
    }
}
