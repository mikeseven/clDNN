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
#include "api/primitives/depth_concatenate.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(depth_concatenate_f32_gpu, test01) {
    //  Input count : 2
    //  Input1 : 2x 1x1 x 2
    //  Input2 : 2x 1x1 x 3
    //
    //  Input1:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //
    //  Input2:
    //  1    0.1  :f0
    //  0.3 -0.5  :f1
    //  0   -0.2  :f2
    //
    //  Output:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //  1    0.1  :f2
    //  0.3 -0.5  :f3
    //  0   -0.2  :f4
    //

    engine engine;
    auto input1 = memory::allocate(engine, {data_types::f32, tensor(format::yxfb, { 1,1,2,2 })});
    auto input2 = memory::allocate(engine, { data_types::f32, tensor(format::yxfb,{ 1,1,3,2 })});

    set_values(input1, { 0.5f, 0.7f, 0.2f, 0.4f });
    set_values(input2, { 1.0f, 0.1f, 0.3f, -0.5f, 0.0f, -0.2f });

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(depth_concatenate("depth1", { "input1", "input2" }));

    network network(engine, topology);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "depth1");

    auto output = outputs.at("depth1").get_memory();

    auto output_ptr = output.pointer<float>();
    EXPECT_FLOAT_EQ(0.5f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.7f, output_ptr[1]);
    EXPECT_FLOAT_EQ(0.2f, output_ptr[2]);
    EXPECT_FLOAT_EQ(0.4f, output_ptr[3]);
    EXPECT_FLOAT_EQ(1.0f, output_ptr[4]);
    EXPECT_FLOAT_EQ(0.1f, output_ptr[5]);
    EXPECT_FLOAT_EQ(0.3f, output_ptr[6]);
    EXPECT_FLOAT_EQ(-0.5f, output_ptr[7]);
    EXPECT_FLOAT_EQ(0.0f, output_ptr[8]);
    EXPECT_FLOAT_EQ(-0.2f, output_ptr[9]);
}