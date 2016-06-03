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

    auto input_prim   = memory::allocate({ engine::reference, memory::format::xb_f32,{ input_b , {{input_x }}, {1} } });
    auto output_prim  = memory::allocate({ engine::reference, memory::format::xb_f32,{ output_b, {{output_x}}, {1} } });
    auto weights_prim = memory::allocate({ engine::reference, memory::format::oi_f32,{ 1       , {{1}},             {weight_x,weight_y} } });
    auto bias_prim    = memory::allocate({ engine::reference, memory::format::x_f32, { 1,        {{output_x}}, {1} } });
    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim,  bias_prim});

    set_values(input_prim  , {-0.5f, 2.0f, 0.5f});
    set_values(weights_prim, {1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f});
    set_values(bias_prim   , {1.0f, 2.0f, 3.0f, 4.0f});

    execute({full_con_prim}).wait();

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

    auto input_prim   = memory::allocate({ engine::reference, memory::format::xb_f32,{ input_b , {{input_x }}, 1}});
    auto output_prim  = memory::allocate({ engine::reference, memory::format::xb_f32,{ output_b, {{output_x}}, 1}});
    auto weights_prim = memory::allocate({ engine::reference, memory::format::oi_f32,{ 1       , {{1}},             {weight_x,weight_y} } });
    auto bias_prim    = memory::allocate({ engine::reference, memory::format::x_f32, { 1,        {{output_x}}, 1} });
    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim, bias_prim });

    set_values(input_prim  , {-0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f});
    set_values(weights_prim, {1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f});
    set_values(bias_prim   , {1.0f, 2.0f, 3.0f, 4.0f});

    execute({full_con_prim}).wait();

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

    auto input_prim   = memory::allocate({ engine::reference, memory::format:: x_f32, {1       , {{input_x }}, 1 } });
    auto output_prim  = memory::allocate({ engine::reference, memory::format:: x_f32, {1       , {{output_x}}, 1 } });
    auto weights_prim = memory::allocate({ engine::reference, memory::format::oi_f32,{ 1       , {{1}},             {weight_x,weight_y} } });
    auto bias_prim    = memory::allocate({ engine::reference, memory::format:: x_f32, {1,        {{output_x}}, 1 } });

    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim, bias_prim });

    auto& output_memory  = output_prim.as<const memory&>();

    set_values(input_prim  , {-0.5f, 2.0f, 0.5f});
    set_values(weights_prim, {1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f});
    set_values(bias_prim   , {1.0f, 2.0f, 3.0f, 4.0f});

    execute({full_con_prim}).wait();

    EXPECT_EQ( 2.5f,  get_value<float>(output_memory,0));
    EXPECT_EQ( 2.75f, get_value<float>(output_memory,1));
    EXPECT_EQ( 0.75f, get_value<float>(output_memory,2));
    EXPECT_EQ( 7.0f,  get_value<float>(output_memory,3));
}






static void fillrand(primitive& mr)
{
    auto& mem = mr.as<const neural::memory&>();
    auto it = static_cast<float*>(mem.pointer);
    auto size = mem.argument.size.spatial[0] * mem.argument.size.batch[0];
    if (mem.argument.size.feature.size() != 1)
        size = mem.argument.size.feature[0] * mem.argument.size.feature[1];
    for (int i = 0; i < size; i++)
        *it++ = (rand() % 20 - 10) / 2.0f;
};

static void fillzero(primitive& mr)
{
    auto& mem = mr.as<const neural::memory&>();
    auto it = static_cast<float*>(mem.pointer);
    auto size = mem.argument.size.spatial[0] * mem.argument.size.batch[0];
    if (mem.argument.size.feature.size() != 1)
        size = mem.argument.size.feature[0] * mem.argument.size.feature[1];
    for (int i = 0; i < size; i++)
        *it++ = 5.0f;
};

static void TransposeMatrix(primitive& dst, primitive& src)
{
    auto& mem_src = src.as<const neural::memory&>();
    auto& mem_dst = dst.as<const neural::memory&>();
    auto size = mem_src.argument.size.feature[0] * mem_dst.argument.size.feature[1];
    auto width = mem_src.argument.size.feature[0];
    auto length = mem_dst.argument.size.feature[1];

    auto memp_src = static_cast<float*>(mem_src.pointer);
    auto memp_dst = static_cast<float*>(mem_dst.pointer);

    for (int i = 0; i < size; i++)
    {
        auto x = i / length;
        auto y = i % length;
        memp_dst[x + y * width] = memp_src[i];
    }
};

TEST(fully_connected_avx2_batch1, x_f32) 
{
    const uint32_t output_x = 129,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_x = output_x, weight_y = input_x;  // size of whole weights buffer

    auto input_prim     = memory::allocate({ engine::reference, memory::format:: x_f32,{ 1       ,{ { input_x } }, 1 }});
    auto output_prim    = memory::allocate({ engine::reference, memory::format::x_f32,{ 1       ,{ { output_x } }, 1 }});
    auto output_prim_ref= memory::allocate({ engine::reference, memory::format:: x_f32,{ 1       ,{ { output_x } }, 1 }});
    auto weights_prim = memory::allocate({ engine::reference, memory::format::oi_f32,{ 1,{{1}}, { weight_x, weight_y } }});
    auto weights_prim_tr = memory::allocate({ engine::reference, memory::format::io_f32,{ 1,{{1}}, { weight_x, weight_y } }});
    auto bias_prim      = memory::allocate({ engine::reference, memory::format:: x_f32,{ 1,       { { output_x } }, 1 }});

    auto full_con_prim_ref = fully_connected::create({ engine::reference, output_prim_ref, input_prim, weights_prim, bias_prim });

    auto& output_memory_ref = output_prim_ref.as<const memory&>();

    srand(0);
    fillrand(input_prim);
    fillrand(weights_prim);
    fillrand(bias_prim);
  
    //TransposeMatrix(weights_prim_tr, weights_prim);
    auto reorder = reorder::create({engine::reference, weights_prim, weights_prim_tr});
    


    execute({ full_con_prim_ref }).wait();

    auto full_con_prim = fully_connected::create({ engine::cpu, output_prim, input_prim, weights_prim_tr, bias_prim });
    execute({ reorder}).wait();
    execute({ full_con_prim }).wait();

    auto& output_memory = output_prim.as<const memory&>();

    auto it_ref = static_cast<float*>(output_memory_ref.pointer);
    auto it = static_cast<float*>(output_memory.pointer);

    for (int i = 0; i < output_memory_ref.count(); i++)
        EXPECT_EQ(true, tests::are_equal(get_value<float>(output_memory_ref, i), get_value<float>(output_memory, i))) << " at index " << i << "\n";
}



TEST(fully_connected_avx2_batch8, x_f32) 
{
    const uint32_t output_x = 13*8, output_b = 8,              // size of whole output buffer
        input_x = 3*10, input_b = 8,    // size of whole input buffer
        weight_x = output_x, weight_y = input_x;  // size of whole weights buffer

    auto input_prim     = memory::allocate({ engine::reference, memory::format:: x_f32,{ input_b       ,{ { input_x } }, 1 }});

    auto output_prim_ref= memory::allocate({ engine::reference, memory::format::x_f32,{ output_b       ,{ { output_x } }, 1 }});
    auto output_prim    = memory::allocate({ engine::reference, memory::format::x_f32,{ output_b       ,{ { output_x } }, 1 }});

    auto weights_prim   = memory::allocate({ engine::reference, memory::format::oi_f32,{ 1, {{1}}, { weight_x, weight_y } }});
    auto weights_prim_tr = memory::allocate({ engine::reference, memory::format::io_i13_f32, { 1,{{1}}, { weight_x, weight_y } }});
    auto bias_prim      = memory::allocate({ engine::reference, memory::format:: x_f32,{ 1,       { { output_x } }, 1 }});

    auto& output_memory_ref = output_prim_ref.as<const memory&>();
    auto& output_memory = output_prim.as<const memory&>();
    auto it_ref = static_cast<float*>(output_memory_ref.pointer);
    auto it = static_cast<float*>(output_memory.pointer);

    auto w1 = static_cast<float*>( (weights_prim.as<const memory&>()).pointer);
    auto w2 = static_cast<float*>( (weights_prim_tr.as<const memory&>()).pointer);
     
    auto full_con_prim_ref = fully_connected::create({ engine::reference, output_prim_ref , input_prim, weights_prim, bias_prim });
    auto full_con_prim = fully_connected::create({ engine::cpu, output_prim, input_prim, weights_prim_tr, bias_prim });

    srand(0);
    fillrand(input_prim);
    fillrand(weights_prim);
    fillrand(bias_prim);
    auto reorder = reorder::create({engine::reference, weights_prim, weights_prim_tr});

    execute({ reorder}).wait();

    execute({ full_con_prim_ref }).wait();
    execute({ full_con_prim }).wait();

    for (int i = 0; i < output_memory_ref.count(); i++)
        EXPECT_EQ(true, tests::are_equal(get_value<float>(output_memory_ref, i), get_value<float>(output_memory, i))) << " at index " << i << "\n";
}
