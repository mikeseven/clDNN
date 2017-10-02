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
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/assign_patch.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/reorder.hpp>
#include <api/CPP/data.hpp>

using namespace cldnn;
using namespace tests;

TEST(assign_patch_gpu, basic_in1x2x4x2) {
    //  Input  : 1x2x3x2
    //  NN     : 1x1x2x2
    //  Output : 2x2x6x4

    //  Input:
    //  f0:  1    2  -10 
    //  f0:  3    4  -14 
    //  f1:  5    6  -12    
    //  f1:  7    8  -16 
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 3, 2 } });
    auto nn = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("nn", nn));
    topology.add(assign_patch("assign_patch", "input", "nn"));

    set_values(input, {
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f,
        5.f, 6.f, -12.f,
        7.f, 8.f, -16.f,
    });

    set_values(nn, { 0.f, 0.f, 0.f, 0.f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("assign_patch").get_memory();
    auto output_ptr = output.pointer<float>();

    //TODO: verify correctness of this patching
    float answers[24] = {
        1.f, 3.f, 2.f, -10.f,
        -6.f, 6.f, 6.f, -20.f,
        -21.f, 3.f, 4.f, -10.f,
        -9.f, 11.f, 6.f, -12.f,
        0.f, 18.f, 14.f, -20.f,
        -21.f, 7.f, 8.f, -8.f
    };

    for (int j = 0; j < 2; ++j) { //F
        for (int k = 0; k < 3; ++k) { //Y
            for (int l = 0; l < 4; ++l) { //X
                auto output_linear_id = l + k * 4 + j * 3 * 4;
                EXPECT_TRUE(are_equal(answers[output_linear_id], output_ptr[output_linear_id]));
            }
        }
    }
}