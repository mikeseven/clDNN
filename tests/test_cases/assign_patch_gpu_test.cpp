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
    //  Mean   : 3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //
    //  Mean
    //  f0: -3.3333
    //  f1: -0.3583
    //
    //  Variance
    //  f0: 44.9305
    //  f1: 107.0624


    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 4, 3 } });
    auto nn = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("nn", nn));
    topology.add(assign_patch("assign_patch", "input", "nn"));

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 9.f,
        -14.f, -15.f, -16.f, -17.f
    });

    set_values(nn, { 0.f, 0.f, 0.f, 0.f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("assign_patch").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> out;

    for (int j = 0; j < 2; ++j) { //F
            for (int k = 0; k < 4; ++k) { //Y
                for (int l = 0; l < 5; ++l) { //X
                    auto output_linear_id = l + k * 5 + j * 4 * 5;
                    out.push_back(output_ptr[output_linear_id]);
                }
            }
    }
}