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

TEST(mean_subtract_f32, basic_in4x4x2x2) {
    //  Mean   : 2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0       
    //  f0: b0:  3    4  b1:   0.5 -0.5     
    //  f1: b0:  5    6  b1:   1.5  5.2     
    //  f1: b0:  7    8  b1:   12   8       
    //
    //  Mean
    //  f0: 0.5  5 
    //  f0: 15   6
    //  f1: 0.5  2
    //  f1: 8   -0.5
    //
    //  Output:
    //  f0: b0:   0.5 -3    b1:  -0.5 -5       
    //  f0: b0:  -12  -2    b1: -14.5 -6.5     
    //  f1: b0:   4.5  4    b1:   1    3.2     
    //  f1: b0:  -1    8.5  b1:   4    8.5     
    //

    auto input = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 } });
    auto output = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 } });
    auto mean = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ 1,{ 2, 2 }, 2  } });

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f
    });

    set_values(mean, { 0.5f, 0.5f, 5.f, 2.f, 15.f, 8.f, 6.f, -0.5f });

    auto mean_sub = mean_subtract::create({ engine::reference, output, input, mean });

    execute({ mean_sub }).wait();

    float answers[16] = { 0.5f, -0.5f, 4.5f, 1.0f,
                            -3.0f, -5.0f, 4.0f, 3.2f,
                            -12.0f, -14.5f, -1.0f, 4.0f,
                            -2.0f, -6.5f, 8.5f, 8.5f };
    auto output_ptr = output.as<const memory&>().pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}