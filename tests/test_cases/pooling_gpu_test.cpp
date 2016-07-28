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

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz3x3_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 3x3
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 0.0, -1.0,  0.5]
    //
    //  Expected output:
    //  [ 2.0]

    auto input_prim = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 3, 3 }, 1 } });
    auto output_prim = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 1, 1 }, 1 } });
    auto pool_prim = pooling::create({ engine::gpu, pooling::mode::max, output_prim, input_prim,{ 1,{ 1, 1 }, 1 },{ 1,{ 3, 3 }, 1 }, padding::type::zero });

    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f });

    execute({ pool_prim }).wait();

    auto output_ptr = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ(2.0f, get_value<float>(output_ptr, 0));
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 0.0, -1.0,  0.5]
    //
    //  Expected output:
    //  [ 2.0,  1.5]
    //  [ 2.0,  1.5]

    auto input_prim = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 3, 3 }, 1 } });
    auto output_prim = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto pool_prim = pooling::create({ engine::gpu, pooling::mode::max, output_prim, input_prim,{ 1,{ 1, 1 }, 1 },{ 1,{ 2, 2 }, 1 }, padding::type::zero });

    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f });

    execute({ pool_prim }).wait();

    auto output_ptr = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ(2.0f, get_value<float>(output_ptr, 0));
    EXPECT_EQ(1.5f, get_value<float>(output_ptr, 1));
    EXPECT_EQ(2.0f, get_value<float>(output_ptr, 2));
    EXPECT_EQ(1.5f, get_value<float>(output_ptr, 3));
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr2x2_i4x4x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.25,  1.00,  0.50,  0.25]
    //  [ 2.00,  1.50, -0.50, -0.75]
    //  [ 0.00, -1.00,  0.50,  0.25]
    //  [ 0.50, -2.00, -1.50, -2.50]
    //
    //  Expected output:
    //  [ 2.0,  0.5]
    //  [ 0.5,  0.5]

    auto input_prim = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 4, 4 }, 1 } });
    auto output_prim = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto pool_prim = pooling::create({ engine::gpu, pooling::mode::max, output_prim, input_prim,{ 1,{ 2, 2 }, 1 },{ 1,{ 2, 2 }, 1 }, padding::type::zero });

    set_values(input_prim, { -0.25f, 1.00f, 0.50f, 0.25f, 2.00f, 1.50f, -0.50f, -0.75f, 0.00f, -1.00f, 0.50f, 0.25f, 0.50f, -2.00f, -1.50f, -2.50f });

    execute({ pool_prim }).wait();

    auto output_ptr = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ(2.0f, get_value<float>(output_ptr, 0));
    EXPECT_EQ(0.5f, get_value<float>(output_ptr, 1));
    EXPECT_EQ(0.5f, get_value<float>(output_ptr, 2));
    EXPECT_EQ(0.5f, get_value<float>(output_ptr, 3));
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x2x2_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  FM: 0 BATCH: 0       FM: 1 BATCH: 0
    //  [-0.5,  0.5,  0.0]   [-1.5, -0.5,  0.0]
    //  [ 1.0, -1.0, -2.0]   [ 0.0, -1.0,  1.5]
    //  [-1.0, -0.5, -0.5]   [-2.0,  1.0, -0.5]
    //
    //  FM: 0 BATCH: 1       FM: 1 BATCH: 1
    //  [ 0.5,  0.0, -0.5]   [ 0.0,  0.5, -0.5]
    //  [-2.0, -1.0,  1.0]   [ 1.0, -1.0,  0.0]
    //  [-0.5, -1.0,  1.5]   [ 0.5, -0.5,  0.0]
    //
    //  Expected output:
    //  FM: 0 BATCH: 0       FM: 1 BATCH: 0
    //  [ 1.0,  0.5]         [ 0.0,  1.5]
    //  [ 1.0, -0.5]         [ 1.0,  1.5]
    //
    //  FM: 0 BATCH: 1       FM: 1 BATCH: 1
    //  [ 0.5,  1.0]         [ 1.0,  0.5]
    //  [-0.5,  1.5]         [ 1.0,  0.0]

    auto input_prim = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 2,{ 3, 3 }, 2 } });
    auto output_prim = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 } });
    auto pool_prim = pooling::create({ engine::gpu, pooling::mode::max, output_prim, input_prim,{ 1,{ 1, 1 }, 1 },{ 1,{ 2, 2 }, 1 }, padding::type::zero });

    set_values(input_prim, { -0.5f, 0.5f, -1.5f, 0.0f, 0.5f, 0.0f, -0.5f, 0.5f, 0.0f, -0.5f, 0.0f, -0.5f, 1.0f, -2.0f, 0.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -2.0f, 1.0f, 1.5f, 0.0f, -1.0f, -0.5f, -2.0f, 0.5f, -0.5f, -1.0f, 1.0f, -0.5f, -0.5f, 1.5f, -0.5f, 0.0f });

    execute({ pool_prim }).wait();

    auto output_ptr = output_prim.as<const memory&>().pointer<float>();
    EXPECT_EQ(1.0f, get_value<float>(output_ptr, 0)); EXPECT_EQ(0.0f, get_value<float>(output_ptr, 2));
    EXPECT_EQ(0.5f, get_value<float>(output_ptr, 4)); EXPECT_EQ(1.5f, get_value<float>(output_ptr, 6));
    EXPECT_EQ(1.0f, get_value<float>(output_ptr, 8)); EXPECT_EQ(1.0f, get_value<float>(output_ptr, 10));
    EXPECT_EQ(-0.5f, get_value<float>(output_ptr, 12)); EXPECT_EQ(1.5f, get_value<float>(output_ptr, 14));

    EXPECT_EQ(0.5f, get_value<float>(output_ptr, 1)); EXPECT_EQ(1.0f, get_value<float>(output_ptr, 3));
    EXPECT_EQ(1.0f, get_value<float>(output_ptr, 5)); EXPECT_EQ(0.5f, get_value<float>(output_ptr, 7));
    EXPECT_EQ(-0.5f, get_value<float>(output_ptr, 9)); EXPECT_EQ(1.0f, get_value<float>(output_ptr, 11));
    EXPECT_EQ(1.5f, get_value<float>(output_ptr, 13)); EXPECT_EQ(0.0f, get_value<float>(output_ptr, 15));
}