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
#include "multidimensional_counter.h"

using namespace neural;

TEST(fully_connected, xb_f32) {

/*
    Input  : 3x1
    Output : 4x1
    Weights: 4x3

    Input:
    -0.5     2    0.5

    Weights:
     1.5     1    0.5
    -1       0    0.5
     0.5    -0.5 -2
    -0.5     1    1.5

    Output:
     1.5    0.75    -2.25   3

*/

    const uint32_t output_x  = 4, output_b  = 1,  // size of whole output buffer
                   input_x   = 3, input_b   = 1,  // size of whole input buffer
                   weight_x  = 4, weight_y  = 3;  // size of whole weights buffer

    auto input_prim   = memory::create({ engine::cpu, memory::format::xb_f32,{ input_x,  input_b  } , true });
    auto output_prim  = memory::create({ engine::cpu, memory::format::xb_f32,{ output_x, output_b } , true });
    auto weights_prim = memory::create({ engine::cpu, memory::format::xb_f32,{ weight_x, weight_y } , true });
    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim });

    auto& input_memory   = input_prim.as<const memory&>();
    auto& output_memory  = output_prim.as<const memory&>();
    auto& weights_memory = weights_prim.as<const memory&>();

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  2.0f);
    input_memory.set_value<float>(2,  0.5f);

    weights_memory.set_value<float>(0,  1.5f);
    weights_memory.set_value<float>(1,  1.0f);
    weights_memory.set_value<float>(2,  0.5f);
    weights_memory.set_value<float>(3, -1.0f);
    weights_memory.set_value<float>(4,  0.0f);
    weights_memory.set_value<float>(5,  0.5f);
    weights_memory.set_value<float>(6,  0.5f);
    weights_memory.set_value<float>(7, -0.5f);
    weights_memory.set_value<float>(8, -2.0f);
    weights_memory.set_value<float>(9, -0.5f);
    weights_memory.set_value<float>(10, 1.0f);
    weights_memory.set_value<float>(11, 1.5f);

    output_memory.fill<float>(0.0f);

    execute({full_con_prim});

    EXPECT_EQ(1.5f,   output_memory.get_value<float>(0));
    EXPECT_EQ(0.75f,  output_memory.get_value<float>(1));
    EXPECT_EQ(-2.25f, output_memory.get_value<float>(2));
    EXPECT_EQ(3.0f,   output_memory.get_value<float>(3));
}