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

TEST(fully_connected_relu_gpu, xb_f32_batch_1) {
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
    //   1.0,  -2.0,  3.0,  -4.0
    //
    //  Output:
    //   2.5   0      0.75  0

    const uint32_t output_x = 4, output_b = 1,  // size of whole output buffer
        input_x = 3, input_b = 1,  // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto input_prim = memory::allocate({ engine::gpu, memory::format::xb_f32,{ input_b ,{ { input_x } },{ 1 } } });
    auto output_prim = memory::allocate({ engine::gpu, memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate({ engine::gpu, memory::format::xb_f32,{ weight_y,{ { weight_x } },{ 1 } } });
    auto bias_prim = memory::allocate({ engine::gpu, memory::format::x_f32,{ 1,{ { output_x } },{ 1 } } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    auto full_con_relu_prim = fully_connected_relu::create({ engine::gpu, output_prim, input_prim, weights_prim,  bias_prim, 0 });

    execute({ full_con_relu_prim }).wait();

    auto output_ptr = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(0.00f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
}

TEST(fully_connected_relu_gpu, xb_f32_batch_2) {
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
    //   1.0, -2.0, 3.0, -4.0
    //
    //  Output:
    //   2.5    0   0.75   0
    //   4      0   2.75   0

    const uint32_t output_x = 4, output_b = 2,  // size of whole output buffer
        input_x = 3, input_b = 2,  // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto input_prim   = memory::allocate({ engine::gpu, memory::format::xb_f32,{ input_b ,{ { input_x  } }, 1 } });
    auto output_prim  = memory::allocate({ engine::gpu, memory::format::xb_f32,{ output_b,{ { output_x } }, 1 } });
    auto weights_prim = memory::allocate({ engine::gpu, memory::format::xb_f32,{ weight_y,{ { weight_x } }, 1 } });
    auto bias_prim =    memory::allocate({ engine::gpu, memory::format::x_f32, { 1,{ { output_x } }, 1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    auto full_con_rel_prim = fully_connected_relu::create({ engine::gpu, output_prim, input_prim, weights_prim, bias_prim, 0 });

    execute({ full_con_rel_prim }).wait();

    auto output_ptr = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(4.00f, output_ptr[1]);
    EXPECT_EQ(0.00f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
    EXPECT_EQ(0.75f, output_ptr[4]);
    EXPECT_EQ(2.75f, output_ptr[5]);
    EXPECT_EQ(0.00f, output_ptr[6]);
    EXPECT_EQ(0.00f, output_ptr[7]);
}

TEST(fully_connected_relu_gpu, x_f32) {
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
    //   1.0, -2.0, 3.0, -4.0
    //  Output:
    //   2.5   0    0.75  0

    const uint32_t output_x = 4,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto input_prim   = memory::allocate({ engine::gpu, memory::format::x_f32, { 1       ,{ { input_x  } }, 1 } });
    auto output_prim  = memory::allocate({ engine::gpu, memory::format::x_f32, { 1       ,{ { output_x } }, 1 } });
    auto weights_prim = memory::allocate({ engine::gpu, memory::format::xb_f32,{ weight_y,{ { weight_x } }, 1 } });
    auto bias_prim    = memory::allocate({ engine::gpu, memory::format::x_f32, { 1,{ { output_x } }, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    auto full_con_relu_prim = fully_connected_relu::create({ engine::gpu, output_prim, input_prim, weights_prim, bias_prim, 0 });

    execute({ full_con_relu_prim }).wait();

    auto output_ptr = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(0.00f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
}
