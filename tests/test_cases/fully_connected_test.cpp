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
#include "multidimensional_counter.h"
#include "tests/gtest/gtest.h"
#include "test_utils/test_utils.h"
#include "memory_utils.h"

using namespace neural;
using namespace tests;

TEST(fully_connected, xb_f32_batch_1) {
//  Input  : 3x1
//  Output : 4x1
//  Weights: 4x3
//
//  Input:
//  -0.5     2    0.5
//
//  Weights:
//   1.5     1    0.5
//  -1       0    0.5
//   0.5    -0.5 -2
//  -0.5     1    1.5
//
//
//  Biases:
//   1.0, 2.0, 3.0, 4.0
//
//  Output:
//   2.5    2.75    0.75   7

    const uint32_t output_x  = 4, output_b  = 1,  // size of whole output buffer
                   input_x   = 3, input_b   = 1,  // size of whole input buffer
                   weight_x  = 4, weight_y  = 3;  // size of whole weights buffer

    auto input_prim   = memory::create({ engine::reference, memory::format::xb_f32,{ input_b , {{input_x }}, {1} } , true });
    auto output_prim  = memory::create({ engine::reference, memory::format::xb_f32,{ output_b, {{output_x}}, {1} } , true });
    auto weights_prim = memory::create({ engine::reference, memory::format::xb_f32,{ weight_y, {{weight_x}}, {1} } , true });
    auto bias_prim    = memory::create({ engine::reference, memory::format::x_f32, { 1,        {{output_x}}, {1} } , true });
    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim,  bias_prim});

    set_values(input_prim  , {-0.5f, 2.0f, 0.5f});
    set_values(weights_prim, {1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f});
    set_values(bias_prim   , {1.0f, 2.0f, 3.0f, 4.0f});

    execute({full_con_prim}).sync();

    auto& output_memory  = output_prim.as<const memory&>();
    EXPECT_EQ( 2.5f,  get_value<float>(output_memory, 0));
    EXPECT_EQ( 2.75f, get_value<float>(output_memory, 1));
    EXPECT_EQ( 0.75f, get_value<float>(output_memory, 2));
    EXPECT_EQ( 7.0f,  get_value<float>(output_memory, 3));
}

TEST(fully_connected, xb_f32_batch_2) {
//  Input  : 3x2
//  Output : 4x2
//  Weights: 4x3
//
//  Input:
//  -0.5     2    0.5
//   1       1.5  0
//
//  Weights:
//   1.5     1    0.5
//  -1       0    0.5
//   0.5    -0.5 -2
//  -0.5     1    1.5
//
//  Biases:
//   1.0, 2.0, 3.0, 4.0
//
//  Output:
//   2.5    2.75     0.75   7
//   4      1        2.75   5

    const uint32_t output_x  = 4, output_b  = 2,  // size of whole output buffer
                   input_x   = 3, input_b   = 2,  // size of whole input buffer
                   weight_x  = 4, weight_y  = 3;  // size of whole weights buffer

    auto input_prim   = memory::create({ engine::reference, memory::format::xb_f32,{ input_b , {{input_x }}, 1}, true });
    auto output_prim  = memory::create({ engine::reference, memory::format::xb_f32,{ output_b, {{output_x}}, 1}, true });
    auto weights_prim = memory::create({ engine::reference, memory::format::xb_f32,{ weight_y, {{weight_x}}, 1}, true });
    auto bias_prim    = memory::create({ engine::reference, memory::format::x_f32, { 1,        {{output_x}}, 1} , true });
    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim, bias_prim });

    set_values(input_prim  , {-0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f});
    set_values(weights_prim, {1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f});
    set_values(bias_prim   , {1.0f, 2.0f, 3.0f, 4.0f});

    execute({full_con_prim}).sync();

    auto& output_memory  = output_prim.as<const memory&>();
    EXPECT_EQ( 2.5f,  get_value<float>(output_memory,0));
    EXPECT_EQ( 4.0f,  get_value<float>(output_memory,1));
    EXPECT_EQ( 2.75f, get_value<float>(output_memory,2));
    EXPECT_EQ( 1.0f,  get_value<float>(output_memory,3));
    EXPECT_EQ( 0.75f, get_value<float>(output_memory,4));
    EXPECT_EQ( 2.75f, get_value<float>(output_memory,5));
    EXPECT_EQ( 7.0f,  get_value<float>(output_memory,6));
    EXPECT_EQ( 5.0f,  get_value<float>(output_memory,7));
}


TEST(fully_connected, x_f32) {
//  Input  : 3x1
//  Output : 4x1
//  Weights: 4x3
//
//  Input:
//  -0.5     2    0.5
//
//  Weights:
//   1.5     1    0.5
//  -1       0    0.5
//   0.5    -0.5 -2
//  -0.5     1    1.5
//
//  Biases:
//   1.0, 2.0, 3.0, 4.0
//  Output:
//   2.5    2.75    0.75   7

    const uint32_t output_x  = 4,                 // size of whole output buffer
                   input_x   = 3,                 // size of whole input buffer
                   weight_x  = 4, weight_y  = 3;  // size of whole weights buffer

    auto input_prim   = memory::create({ engine::reference, memory::format:: x_f32, {1       , {{input_x }}, 1 } , true });
    auto output_prim  = memory::create({ engine::reference, memory::format:: x_f32, {1       , {{output_x}}, 1 } , true });
    auto weights_prim = memory::create({ engine::reference, memory::format::xb_f32, {weight_y, {{weight_x}}, 1 } , true });
    auto bias_prim    = memory::create({ engine::reference, memory::format:: x_f32, {1,        {{output_x}}, 1 } , true });

    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim, bias_prim });

    auto& output_memory  = output_prim.as<const memory&>();

    set_values(input_prim  , {-0.5f, 2.0f, 0.5f});
    set_values(weights_prim, {1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f});
    set_values(bias_prim   , {1.0f, 2.0f, 3.0f, 4.0f});

    execute({full_con_prim}).sync();

    EXPECT_EQ( 2.5f,  get_value<float>(output_memory,0));
    EXPECT_EQ( 2.75f, get_value<float>(output_memory,1));
    EXPECT_EQ( 0.75f, get_value<float>(output_memory,2));
    EXPECT_EQ( 7.0f,  get_value<float>(output_memory,3));
}

TEST(fully_connected_gpu, xb_f32_batch_2) {
    //  Input  : 3x2
    //  Output : 4x2
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //   1       1.5  0
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //
    //  Output:
    //   2.5    2.75     0.75   7
    //   4      1        2.75   5

    const uint32_t output_x = 4, output_b = 2,  // size of whole output buffer
        input_x = 3, input_b = 2,  // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto input_prim = memory::create({ engine::gpu, memory::format::xb_f32,{ input_b ,{ { input_x } }, 1 }, true });
    auto output_prim = memory::create({ engine::gpu, memory::format::xb_f32,{ output_b,{ { output_x } }, 1 }, true });
    auto weights_prim = memory::create({ engine::gpu, memory::format::xb_f32,{ weight_y,{ { weight_x } }, 1 }, true });
    auto bias_prim = memory::create({ engine::gpu, memory::format::x_f32,{ 1,{ { output_x } }, 1 } , true });
    auto full_con_prim = fully_connected::create({ engine::gpu, output_prim, input_prim, weights_prim, bias_prim });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    execute({ full_con_prim }).sync();

    auto& output_memory = output_prim.as<const memory&>();
    EXPECT_EQ(2.5f, get_value<float>(output_memory, 0));
    EXPECT_EQ(4.0f, get_value<float>(output_memory, 1));
    EXPECT_EQ(2.75f, get_value<float>(output_memory, 2));
    EXPECT_EQ(1.0f, get_value<float>(output_memory, 3));
    EXPECT_EQ(0.75f, get_value<float>(output_memory, 4));
    EXPECT_EQ(2.75f, get_value<float>(output_memory, 5));
    EXPECT_EQ(7.0f, get_value<float>(output_memory, 6));
    EXPECT_EQ(5.0f, get_value<float>(output_memory, 7));
}

TEST(fully_connected_gpu, x_f32) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //  Output:
    //   2.5    2.75    0.75   7

    const uint32_t output_x = 4,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto input_prim = memory::create({ engine::gpu, memory::format::x_f32,{ 1       ,{ { input_x } }, 1 } , true });
    auto output_prim = memory::create({ engine::gpu, memory::format::x_f32,{ 1       ,{ { output_x } }, 1 } , true });
    auto weights_prim = memory::create({ engine::gpu, memory::format::xb_f32,{ weight_y,{ { weight_x } }, 1 } , true });
    auto bias_prim = memory::create({ engine::gpu, memory::format::x_f32,{ 1,{ { output_x } }, 1 } , true });

    auto full_con_prim = fully_connected::create({ engine::gpu, output_prim, input_prim, weights_prim, bias_prim });

    auto& output_memory = output_prim.as<const memory&>();

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    execute({ full_con_prim }).sync();

    EXPECT_EQ(2.5f, get_value<float>(output_memory, 0));
    EXPECT_EQ(2.75f, get_value<float>(output_memory, 1));
    EXPECT_EQ(0.75f, get_value<float>(output_memory, 2));
    EXPECT_EQ(7.0f, get_value<float>(output_memory, 3));
}
