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
#include "api/CPP/reorg_yolo.hpp"
#include "api/CPP/reorder.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/data.hpp>

#include <cmath>
#include <gmock/gmock.h>
#include <limits>

using namespace cldnn;
using namespace tests;
using namespace testing;

TEST(reorg_yolo_gpu_f32, darknet_test)
{
    //  Input               : bfyx:2x2x3x2
    //  Permute order       : { 0,1,3,2 }
    //
    //  Input:
    //  f0: b0:  1    2   -15  b1:   0    0     -15
    //  f0: b0:  3    4   -15  b1:   0.5 -0.5   -15
    //  f1: b0:  5    6   -15  b1:   1.5  5.2   -15
    //  f1: b0:  7    8   -15  b1:   12   8     -15
    //
    //  Input:
    //  f0: b0:  1    3  b1:   0    0.5
    //  f0: b0:  2    4  b1:   0    -0.5
    //  f0: b0:  -15 -15 b1:   -15  -15
    //  f1: b0:  5    7  b1:   1.5  12
    //  f1: b0:  6    8  b1:   5.2   8
    //  f1: b0:  -15 -15 b1:   -15   -15
    //
    extern std::vector<float> yolo_reorg_input;
    extern std::vector<float> yolo_reorg_ref;
    engine engine;
    const auto inpute_size = 64 * 26 * 26;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 64, 26, 26 } });

    set_values(input, yolo_reorg_input);

    topology topology(
        input_layout("input", input.get_layout()),
        reorg_yolo("reorg_yolo", "input", 2));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorg_yolo");

    auto output = outputs.begin()->second.get_memory();

    float epsilon = 0.00001f;
    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < inpute_size; i++)
    {
        EXPECT_NEAR(yolo_reorg_ref[i], output_ptr[i], epsilon);
    }
    
}
