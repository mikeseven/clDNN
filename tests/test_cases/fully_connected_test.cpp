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
#include <gtest/gtest.h>
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

    auto input_prim   = memory::allocate({ engine::reference, memory::format::xb_f32,{ input_b , {{input_x }}, {1} } });
    auto output_prim  = memory::allocate({ engine::reference, memory::format::xb_f32,{ output_b, {{output_x}}, {1} } });
    auto weights_prim = memory::allocate({ engine::reference, memory::format::xb_f32,{ weight_y, {{weight_x}}, {1} } });
    auto bias_prim    = memory::allocate({ engine::reference, memory::format::x_f32, { 1,        {{output_x}}, {1} } });
    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim,  bias_prim});

    set_values(input_prim  , {-0.5f, 2.0f, 0.5f});
    set_values(weights_prim, {1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f});
    set_values(bias_prim   , {1.0f, 2.0f, 3.0f, 4.0f});

    execute({full_con_prim}).wait();

    auto output_ptr  = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ( 2.5f,  get_value<float>(output_ptr, 0));
    EXPECT_EQ( 2.75f, get_value<float>(output_ptr, 1));
    EXPECT_EQ( 0.75f, get_value<float>(output_ptr, 2));
    EXPECT_EQ( 7.0f,  get_value<float>(output_ptr, 3));
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

    auto input_prim   = memory::allocate({ engine::reference, memory::format::xb_f32,{ input_b , {{input_x }}, 1}});
    auto output_prim  = memory::allocate({ engine::reference, memory::format::xb_f32,{ output_b, {{output_x}}, 1}});
    auto weights_prim = memory::allocate({ engine::reference, memory::format::xb_f32,{ weight_y, {{weight_x}}, 1}});
    auto bias_prim    = memory::allocate({ engine::reference, memory::format::x_f32, { 1,        {{output_x}}, 1} });
    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim, bias_prim });

    set_values(input_prim  , {-0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f});
    set_values(weights_prim, {1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f});
    set_values(bias_prim   , {1.0f, 2.0f, 3.0f, 4.0f});

    execute({full_con_prim}).wait();

    auto output_ptr  = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ( 2.5f,  get_value<float>(output_ptr,0));
    EXPECT_EQ( 4.0f,  get_value<float>(output_ptr,1));
    EXPECT_EQ( 2.75f, get_value<float>(output_ptr,2));
    EXPECT_EQ( 1.0f,  get_value<float>(output_ptr,3));
    EXPECT_EQ( 0.75f, get_value<float>(output_ptr,4));
    EXPECT_EQ( 2.75f, get_value<float>(output_ptr,5));
    EXPECT_EQ( 7.0f,  get_value<float>(output_ptr,6));
    EXPECT_EQ( 5.0f,  get_value<float>(output_ptr,7));
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

    auto input_prim   = memory::allocate({ engine::reference, memory::format:: x_f32, {1       , {{input_x }}, 1 } });
    auto output_prim  = memory::allocate({ engine::reference, memory::format:: x_f32, {1       , {{output_x}}, 1 } });
    auto weights_prim = memory::allocate({ engine::reference, memory::format::xb_f32, {weight_y, {{weight_x}}, 1 } });
    auto bias_prim    = memory::allocate({ engine::reference, memory::format:: x_f32, {1,        {{output_x}}, 1 } });

    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim, bias_prim });

    set_values(input_prim  , {-0.5f, 2.0f, 0.5f});
    set_values(weights_prim, {1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f});
    set_values(bias_prim   , {1.0f, 2.0f, 3.0f, 4.0f});

    execute({full_con_prim}).wait();

    auto output_ptr = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ( 2.5f,  get_value<float>(output_ptr,0));
    EXPECT_EQ( 2.75f, get_value<float>(output_ptr,1));
    EXPECT_EQ( 0.75f, get_value<float>(output_ptr,2));
    EXPECT_EQ( 7.0f,  get_value<float>(output_ptr,3));
}

TEST(fully_connected, DISABLED_yxfn_f32) {
    //  Input  : 1x2x1x2 - 1 batch 2 feature maps of size 2x1
    //  Output : 2x1 - 2 batches 1 neuron each
    //  Weights: 2x2x1x2 - 2 neurons with weights of 2 feature maps of size 2x1
    //
    //  Input:
    //   1  -2      f0: b0
    //   3  -4      f1: b0

    //  Weights:
    //   1  -1      n0: fm0  
    //   2   0      n0: fm1
    //   3   4      n1: fm0
    //   0.5 5      n1: fm1
    //
    //  Biases:
    //   1.0 -5
    //
    //  Output:
    //   10  -28.5

    auto input_prim = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ 1,{ { 2, 1 } }, 2 } });
    auto output_prim = memory::allocate({ engine::reference, memory::format::xb_f32,{ 2 ,{ { 1 } }, 1 } });
    auto weights_prim = memory::allocate({ engine::reference, memory::format::bfyx_f32,{ 2,{ { 2, 1 } }, 2 } });
    auto bias_prim = memory::allocate({ engine::reference, memory::format::x_f32,{ 1,{ { 2 } }, 1 } });

    set_values(input_prim, { 1.f, 3.f, -2.f, -4.f });
    set_values(weights_prim, { 1.f, -1.f, 2.0f, 0.f, 3.0f, 4.0f, 0.5f, 5.0f });
    set_values(bias_prim, { 1.0f, -5.0f });

    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim, bias_prim });

    execute({ full_con_prim }).wait();

    auto output_ptr = output_prim.as<const memory &>().pointer<float>();
    EXPECT_EQ(10, output_ptr[0]);
    EXPECT_EQ(-28.5, output_ptr[1]);
}