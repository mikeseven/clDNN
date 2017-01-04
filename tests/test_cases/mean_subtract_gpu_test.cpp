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
#include <gtest/gtest.h>
#include <api/memory.hpp>
#include <api/primitives/input_layout.hpp>
#include "api/primitives/mean_substract.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(mean_subtract_gpu_f32, basic_in4x4x2x2) {
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

    engine engine;

    auto input  = memory::allocate(engine, { data_types::f32, { format::yxfb, { 2, 2, 2, 2 } } });
    auto mean = memory::allocate(engine, { data_types::f32, { format::bfyx, { 1, 2, 2 , 2 } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("mean", mean.get_layout()));
    topology.add(mean_substract("mean_substract", "input", "mean"));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f
    });

    set_values(mean, { 0.5f, 5.f, 15.f, 6.f, 0.5f, 2.f, 8.f, -0.5f });

    network network(engine, topology);
    
    network.set_input_data("input", input);
    network.set_input_data("mean", mean);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs.begin()->first, "mean_substract");

    auto output = outputs.at("mean_substract").get_memory();

    float answers[16] = { 0.5f, -0.5f, 4.5f, 1.0f,
                            -3.0f, -5.0f, 4.0f, 3.2f,
                            -12.0f, -14.5f, -1.0f, 4.0f,
                            -2.0f, -6.5f, 8.5f, 8.5f };
    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}