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
#include "test_utils/test_utils.h"
#include "api/neural.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"
#include <random>


using namespace neural;
using namespace tests;

TEST(convolution_relu_gpu, trivial_convolution_relu) {

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

    auto conv_relu = convolution_relu::create({ engine::gpu, output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero, 0 });
    execute({ conv_relu }).wait();

    auto& output_memory = output.as<const memory&>();
    EXPECT_FLOAT_EQ(4.0f, get_value<float>(output_memory, 0));
    EXPECT_FLOAT_EQ(0.0f, get_value<float>(output_memory, 1));
    EXPECT_FLOAT_EQ(2.0f, get_value<float>(output_memory, 2));
    EXPECT_FLOAT_EQ(5.0f, get_value<float>(output_memory, 3));
}