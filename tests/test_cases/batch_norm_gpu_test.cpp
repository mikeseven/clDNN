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
#include "api/primitives/batch_norm.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

uint32_t max_input_size = 16384; //2^14
uint32_t possible_input_sizes[] = { 1, 2, 4, 8, 16, 32, 64 };

TEST(batch_normalization_gpu, basic_in2x3x2x2_no_global_stats) {
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 2 } } });
    auto mean = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 1 } } });
    auto variance = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 1 } } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("mean", mean.get_layout()));
    topology.add(input_layout("variance", variance.get_layout()));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", false, epsilon));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("mean", mean);
    network.set_input_data("variance", variance);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;
        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[j*2 + i*2 + l*3 + k];
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;
        
        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}

TEST(batch_normalization_gpu, basic_in2x3x2x2_use_global_stats) {
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   8     -17
    //
    //  Mean
    //  f0: 0.5     1       -10.5 
    //  f0: 1.75    2.25    -14.5
    //  f1: 3.25    5.6     -12.5
    //  f1: 9.5     8       -16.5
    //
    //  Variance
    //  f0: 0.25     1       0.25 
    //  f0: 1.56     5.06    0.25
    //  f1: 3.06     0.16    0.25
    //  f1: 6.25     0       0.25

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 2 } } });
    auto mean = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 1 } } });
    auto variance = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 1 } } });

    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("mean", mean.get_layout()));
    topology.add(input_layout("variance", variance.get_layout()));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", true, epsilon));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f
    });

    set_values(mean, { 0.5f, 3.25f, 1.f, 5.6f, -10.5f, -12.5f, 1.75f, 9.5f, 2.25f, 8.f, -14.5f, -16.5f });
    set_values(variance, { 0.25f, 3.06f, 1.f, 0.16f, 0.25f, 0.25f, 1.56f, 6.25f, 5.06f, 0.f, 0.25f, 0.25f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("mean", mean);
    network.set_input_data("variance", variance);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < 2; ++j) { //F
        float sum = 0, var = 0;
        for (int i = 0; i < 2; ++i) { //B
            for (int k = 0; k < 2; ++k) { //Y
                for (int l = 0; l < 3; ++l) { //X
                    float data = output_ptr[j * 2 + i * 2 + l * 3 + k];
                    sum += data;
                    var += data * data;
                }
            }
        }
        sum /= 2 * 3 * 2;
        var /= 2 * 3 * 2;

        EXPECT_NEAR(sum, 0, 1e-03F);
        EXPECT_NEAR(var, 1, 1e-03F);
    }
}