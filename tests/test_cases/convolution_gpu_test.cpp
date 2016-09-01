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
#include <gtest/gtest.h>
#include "test_utils/test_utils.h"
#include "memory_utils.h"
#include <algorithm>
#include <thread>

#include "multidimensional_counter.h"//todo remove

using namespace neural;
using namespace tests;

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //
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

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 4, 4 }, 1 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    auto conv = convolution::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(8.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.5f, output_ptr[1]);
    EXPECT_FLOAT_EQ(6.0f, output_ptr[2]);
    EXPECT_FLOAT_EQ(9.0f, output_ptr[3]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad) {
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
    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 2,{ 2, 2 }, 1 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 2,{ 1, 1 }, 1 } });
    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, { 0.5f, 2.3f, 1.5f, -0.4f, 2.0f, 1.0f, -4.0f, 3.0f });
    set_values(weights, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases, { -1.0f });

    auto conv = convolution::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(3.65f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.36f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz2x1x2x1_in1x2x1_nopad) {
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

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 2, 1 }, 1 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 2 } });
    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1 ,{ 2, 1 },{ 2, 1 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1 ,{ { 2 } }, 1 } });

    set_values(input, { 1.0f, 2.0f });
    set_values(weights, { 1.0f, 2.0f, -1.0f, -2.0f });
    set_values(biases, { 0.1f, -0.2f });

    auto conv = convolution::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 5, 5 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(5.1f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.2f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz3x2x2x1_in2x2x1_nopad) {
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

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 2, 1 }, 2 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 3 } });
    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1 ,{ 2, 1 },{ 3, 2 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1 ,{ { 3 } }, 1 } });

    set_values(input, { 1.0f, 3.0f, 2.0f, 4.0f });
    set_values(weights, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });
    set_values(biases, { -5.0f, -6.0f, -7.0f });

    auto conv = convolution::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 5, 5 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(25.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(64.0f, output_ptr[1]);
    EXPECT_FLOAT_EQ(103.0f, output_ptr[2]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2x1x3_wstr2x2_in2x2x1x1_nopad) {
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

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 3 } });
    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1 ,{ 2, 2 },{ 3, 1 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1 ,{ { 3 } }, 1 } });

    set_values(input, { -2.3f, -0.1f, 3.1f, 1.9f });
    set_values(weights, { -1.1f, 1.5f, 0.5f, -0.5f, 0.1f, 0.2f, 0.4f, 0.7f, 2.0f, -1.0f, 2.5f, -1.5f });
    set_values(biases, { 0.1f, -0.2f, 0.3f });

    auto conv = convolution::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_TRUE(are_equal(3.08f, output_ptr[0]));
    EXPECT_TRUE(are_equal(2.12f, output_ptr[1]));
    EXPECT_TRUE(are_equal(0.7f,  output_ptr[2]));
}

TEST(convolution_f32_fw_gpu, wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
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
    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 1, 1 }, 1 } });
    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 3, 3 },{ 1, 1 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    auto conv = convolution::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });
    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(12.25f, output_ptr[0]);
}

TEST(convolution_f32_fw_gpu, offsets_wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
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
    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1 ,{ 3, 3 },{ 1, 1 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1 ,{ { 1 } } , 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    auto conv = convolution::create({ engine::gpu,
        output,
        { 0,{ 1, 1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        { input, weights, biases },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 2,  2 }, 1 },
        padding::zero });
    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(2.0f, output_ptr[3]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_split2) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4x2
    //  Output : 2x2x2
    //
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

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 4, 4 }, 2 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 2, 2 }, 2 } });
    auto weights1 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases1 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });
    auto weights2 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases2 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

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
        engine::gpu,
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2) {
    //  2x Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x4x4x2
    //  Output : 2x2x2x2
    //
    //  Input:
    //  f0b0: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f0b1: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f1b0:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
    //  f1b1:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
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
    //   8  8 3.65 3.65 0.5  0.5 -5.36 -5.36
    //   6  6 3.65 3.65 9    9   -5.36 -5.36

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 2,{ 4, 4 }, 2 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 } });
    auto weights1 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases1 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });
    auto weights2 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases2 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, {
       -0.5f, -0.5f,  0.5f,  0.5f,  1.0f,  1.0f,  1.5f,  1.5f,  0.5f,  0.5f,  2.3f,  2.3f,  2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, -0.5f, -0.5f, -4.0f, -4.0f,  0.0f,  0.0f,  1.0f,  1.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  1.5f,  1.5f, -1.0f, -1.0f,  2.3f,  2.3f,  1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f,  2.0f,  2.0f, -4.0f, -4.0f,  1.5f,  1.5f,  1.0f,  1.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases1, { 2.0f });
    set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases2, { -1.0f });

    auto conv = convolution::create({
        engine::gpu,
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 8));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 9));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 10));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 11));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 12));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 13));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 14));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 15));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x4x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x4
    //  Output : 1x1x4
    //
    //  Input:
    //  f0:  1.5
    //  f1:  0.5
    //
    //  f2:  0.0
    //  f3: -0.5
    //
    //
    //  Filter1:
    //  -2 -0.5  ofm=0
    //   1  2    ofm=1 
    //  Bias1:
    //   1  5
    //
    //  Filter2:
    //   4  1.5  ofm=0
    //   2  0.5  ofm=1
    //
    //  Bias2:
    //  -1  2.5
    //
    //  Output:
    //  -2.25  
    //   7.5
    //
    //  -1.75
    //   2.25
    

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto weights1 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 2, 2 } } });
    auto biases1 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 2 } } , 1 } });
    auto weights2 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 2, 2 } } });
    auto biases2 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 2 } } , 1 } });

    set_values(input, {
       1.5f, 0.5f, 0.0f, -0.5f
    });
    set_values(weights1, { -2.0f, -0.5f, 1.0f, 2.0f });
    set_values(biases1, { 1.0f, 5.0f });
    set_values(weights2, { 4.0f, 1.5f, 2.0f, 0.5f });
    set_values(biases2, { -1.0f, 2.5f });

    auto conv = convolution::create({
        engine::gpu,
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(-2.25f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(7.5f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(-1.75f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(2.25f, get_value<float>(output_ptr, 3));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x2x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x2
    //  Output : 1x1x4
    //
    //  Input:
    //  f0:  1.5
    //
    //  f1:  0.5
    //
    //  Filter1:
    //  -2  ofm=0
    //   1  ofm=1 
    //  Bias1:
    //   1  5
    //
    //  Filter2:
    //   4  ofm=0
    //   2  ofm=1
    //
    //  Bias2:
    //  -1  2.5
    //
    //  Output:
    //  -2  
    //   6.5
    //
    //   1
    //   3.5


    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 1, 1 }, 2 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto weights1 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 2, 1 } } });
    auto biases1 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 2 } } , 1 } });
    auto weights2 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 2, 1 } } });
    auto biases2 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 2 } } , 1 } });

    set_values(input, {
        1.5f, 0.5f
    });
    set_values(weights1, { -2.0f, 1.0f });
    set_values(biases1, { 1.0f, 5.0f });
    set_values(weights2, { 4.0f, 2.0f });
    set_values(biases2, { -1.0f, 2.5f });

    auto conv = convolution::create({
        engine::gpu,
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(-2.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(6.5f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(1.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(3.5f, get_value<float>(output_ptr, 3));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x4x1_filter_1x3x2x1x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x4
    //  Output : 1x1x6
    //
    //  Input:
    //  f0:  1.5
    //  f1:  0.5
    //
    //  f2:  2
    //  f3: -1.0
    //
    //  Filter1:
    //  -2   1   ofm=0
    //   1   3   ofm=1
    //   0.5 8   ofm=2
    //  Bias1:
    //   1   5   3
    //
    //  Filter2:
    //   4  -4   ofm=0
    //   2   0.5 ofm=1
    //  -0.5 3   ofm=2
    //
    //  Bias2:
    //  -1   2.5 2
    //
    //  Output:
    //  -1.5  
    //   8
    //   7.75
    //
    //   11
    //   6
    //  -2


    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 1, 1 }, 6 } });
    auto weights1 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 3, 2 } } });
    auto biases1 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 3 } } , 1 } });
    auto weights2 = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 3, 2 } } });
    auto biases2 = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 3 } } , 1 } });

    set_values(input, {
        1.5f, 0.5f, 2.0f, -1.0f
    });
    set_values(weights1, { -2.0f, 1.0f, 1.0f, 3.0f, 0.5f, 8.0f });
    set_values(biases1, { 1.0f, 5.0f, 3.0f });
    set_values(weights2, { 4.0f, -4.0f, 2.0f, 0.5f, -0.5f, 3.0f });
    set_values(biases2, { -1.0f, 2.5f, 2.0f });

    auto conv = convolution::create({
        engine::gpu,
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(-1.5f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(7.75f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(11.0f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(-2.0f, get_value<float>(output_ptr, 5));

}

TEST(convolution_gpu, trivial_convolution_relu) {

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
    //  -2
    //
    //  Output:
    //  4  0.0
    //  2  5

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 4, 4 }, 1 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 1 } }, 1 } });

    set_values(input, {
        -0.5f,  1.0f,  0.5f,  2.0f,
        1.5f, -0.5f,  0.0f, -1.0f,
        0.5f,  0.5f, -1.0f,  1.0f,
        0.5f,  2.0f,  1.5f, -0.5f
    });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { -2.0f });

    auto conv_relu = convolution::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero, 1, true, 0 });
    execute({ conv_relu }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(4.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(0.0f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(2.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(5.0f, get_value<float>(output_ptr, 3));
}

/*TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad_reorder) {
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
    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 2,{ 2, 2 }, 1 } });
    auto input2 = memory::allocate({ engine::gpu, memory::format::bfyx_f32,{ 2,{ 2, 2 }, 1 } });

    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 2,{ 1, 1 }, 1 } });
    auto output2 = memory::allocate({ engine::gpu, memory::format::bfyx_f32,{ 2,{ 1, 1 }, 1 } });
    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, { 0.5f, 2.3f, 1.5f, -0.4f, 2.0f, 1.0f, -4.0f, 3.0f });
    set_values(weights, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases, { -1.0f });

    auto input_yxfb_f32_to_bfyx_f32 = reorder::create({ engine::cpu, input, input2 });
    execute({ input_yxfb_f32_to_bfyx_f32 }).wait();

    auto conv = convolution::create({ engine::gpu, output2,{ input2, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });
    execute({ conv }).wait();

    auto output_bfyx_f32_to_yxfb_f32 = reorder::create({ engine::cpu, output2, output });
    execute({ output_bfyx_f32_to_yxfb_f32 }).wait();

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_memory, 0));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_memory, 1));
}*/

TEST(convolution_f32_fw_gpu, DISABLED_basic_ofm_wsiz3x2x2x1_in2x2x1_nopad_reorder) {
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

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 2, 1 }, 2 } });
    auto input2 = memory::allocate({ engine::gpu, memory::format::bfyx_f32,{ 1 ,{ 2, 1 }, 2 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 3 } });
    auto output2 = memory::allocate({ engine::gpu, memory::format::bfyx_f32,{ 1 ,{ 1, 1 }, 3 } });

    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1 ,{ 2, 1 },{ 3, 2 } } });
    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1 ,{ { 3 } }, 1 } });

    set_values(input, { 1.0f, 3.0f, 2.0f, 4.0f });
    set_values(weights, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });
    set_values(biases, { -5.0f, -6.0f, -7.0f });

    auto input_yxfb_f32_to_bfyx_f32 = reorder::create({ engine::cpu, input, input2 });
    execute({ input_yxfb_f32_to_bfyx_f32 }).wait();

    auto conv = convolution::create({ engine::gpu, output2,{ input2, weights, biases },{ 1,{ 5, 5 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_bfyx_f32_to_yxfb_f32 = reorder::create({ engine::cpu, output2, output });
    execute({ output_bfyx_f32_to_yxfb_f32 }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(25.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(64.0f, output_ptr[1]);
    EXPECT_FLOAT_EQ(103.0f, output_ptr[2]);
}

/*unsigned int size_x = 256;
unsigned int size_y = 256;
unsigned int batch_count = 256;

TEST(convolution_f32_fw_gpu, performance_test_yxfb_f32) {
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

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ batch_count ,{ size_x, size_y }, 2 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ batch_count ,{ size_x/2, size_y/2 }, 4 } });

    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1 ,{ 2, 2 },{ 4, 2 } } });

    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1 ,{ { 4 } }, 1 } });

    std::vector<float> _inputs;
    for (unsigned int i = 0; i < size_x * size_y * batch_count * 2; i++)
    {
        _inputs.push_back((float)(i % 100));
    }

    set_values(input, _inputs);

    std::vector<float> _weights;
    for (int i = 0; i < 32; i++)
    {
        _weights.push_back((float)i);
    }
    set_values(weights, _weights);
    set_values(biases, { -1.0f, -2.0f, -3.0f, -4.0f });

    auto conv = convolution::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto& output_memory = output.as<const memory&>();
    //EXPECT_FLOAT_EQ(25.0f, get_value<float>(output_memory, 0));
    //EXPECT_FLOAT_EQ(64.0f, get_value<float>(output_memory, 1));
    //EXPECT_FLOAT_EQ(103.0f, get_value<float>(output_memory, 2));
}

TEST(convolution_f32_fw_gpu, performance_test_bfyx_f32) {
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

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ batch_count ,{ size_x, size_y }, 2 } });
    auto input2 = memory::allocate({ engine::gpu, memory::format::bfyx_f32,{ batch_count ,{ size_x, size_y }, 2 } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ batch_count ,{ size_x/2, size_y/2 }, 4 } });
    auto output2 = memory::allocate({ engine::gpu, memory::format::bfyx_f32,{ batch_count ,{ size_x/2, size_y/2 }, 4 } });

    auto weights = memory::allocate({ engine::gpu, memory::format::oiyx_f32,{ 1 ,{ 2, 2 },{ 4, 2 } } });

    auto biases = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1 ,{ { 4 } }, 1 } });

    std::vector<float> _inputs;
    for (unsigned int i = 0; i < size_x * size_y * batch_count * 2; i++)
    {
        _inputs.push_back((float)(i % 100));
    }

    set_values(input, _inputs);

    std::vector<float> _weights;
    for (int i = 0; i < 32; i++)
    {
        _weights.push_back((float)i);
    }
    set_values(weights, _weights);
    set_values(biases, { -1.0f, -2.0f, -3.0f, -4.0f });

    //auto input_yxfb_f32_to_bfyx_f32 = reorder::create({ engine::cpu, input, input2 });
    //execute({ input_yxfb_f32_to_bfyx_f32 }).wait();

    auto conv = convolution::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });

    execute({ conv }).wait();

   // auto output_bfyx_f32_to_yxfb_f32 = reorder::create({ engine::cpu, output2, output });
    //execute({ output_bfyx_f32_to_yxfb_f32 }).wait();

    auto& output_memory = output.as<const memory&>();
    //EXPECT_FLOAT_EQ(25.0f, get_value<float>(output_memory, 0));
    //EXPECT_FLOAT_EQ(64.0f, get_value<float>(output_memory, 1));
    //EXPECT_FLOAT_EQ(103.0f, get_value<float>(output_memory, 2));
}*/