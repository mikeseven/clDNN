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

TEST(reorder_gpu_f32, basic_subtract_value) {
    //  Values_to_subtract  : 2
    //  Input               : 2x2x2x2
    //  Output              : 2x2x2x2
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  subtract values
    //  f0: 0.5
    //  f1: 2.5
    //
    //  Output:
    //  b0 f0:  0.5  1.5
    //  b0 f0:  2.5  3.5
    //
    //  b0 f1:  2.5  3.5
    //  b0 f1:  4.5  5.5
    //
    //  b1 f0: -0.5 -0.5
    //  b1 f0:  0.0 -1.0
    //
    //  b1 f1: -1.0  2.7
    //  b1 f1:  9.5  5.5
    //

    auto input = memory::allocate({  memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 } });
    auto output = memory::allocate({  memory::format::bfyx_f32,{ 2,{ 2, 2 }, 2 } });
    std::vector<float> subtract_val = { 0.5, 2.5 };

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    auto reorder = reorder::create({  output, input, subtract_val, true });

    execute({ reorder }).wait();

    float answers[16] = { 0.5f, 1.5f,
                          2.5f, 3.5f,

                          2.5f, 3.5f,
                          4.5f, 5.5f,

                         -0.5f, -0.5f,
                          0.0f, -1.0f,

                         -1.0f,  2.7f,
                          9.5f,  5.5f
    };

    auto output_ptr = output.as<const memory&>().pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}