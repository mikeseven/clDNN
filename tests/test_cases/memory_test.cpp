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
#include <api/CPP/engine.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/activation.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/concatenation.hpp>

#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

#if 0
TEST(memory_tests, DISABLED_execution_loop)
{
    engine eng;

    memory in = memory::allocate(eng, layout{ data_types::f32, format::bfyx, { 1, 1, 1000, 1000 } });

    topology tpl{
        input_layout("in", in.get_layout()),
        activation("out", "in", activation_linear)
    };

    network net(eng, tpl);
    
    while (true)
    {
        net.set_input_data("in", in);
        net.execute();
    }
}

TEST(memory_tests, DISABLED_network_creation_loop)
{
    engine eng;

    memory in = memory::allocate(eng, layout{ data_types::f32, format::bfyx,{ 1, 1, 1000, 1000 } });

    topology tpl{
        input_layout("in", in.get_layout()),
        activation("out", "in", activation_linear)
    };

    while (true)
    {
        network net(eng, tpl);
    }
}
#endif
TEST(memory_pool, basic_non_padded_relu_pipe) {
    // 5 relu's of size 1x4x1x1
    engine engine;
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu", "input", activation_relu));
    topology.add(activation("relu1", "relu", activation_relu));
    topology.add(activation("relu2", "relu1", activation_relu));
    topology.add(activation("relu3", "relu2", activation_relu));
    topology.add(activation("relu4", "relu3", activation_relu));
    topology.add(activation("relu5", "relu4", activation_relu));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(engine.get_total_device_memory_size(), (uint64_t) 80);
 }


TEST(memory_pool, basic_non_padded_relu_and_pooling_pipe) {
    // uncomment this line to disable memory pool
    /*engine_configuration cfg{ false, false, false, std::string(), std::string(), true, std::string(),std::string(), 0, false };
    engine engine{ cfg };*/
    engine engine;
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 4;
    auto y_size = 4;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu", "input", activation_relu));
    topology.add(activation("relu1", "relu", activation_relu));
    topology.add(pooling("pool1", "relu1",pooling_mode::max, { 1,1,3,3 }, { 1,1,2,2 }));
    topology.add(activation("relu2", "pool1", activation_relu));
    topology.add(activation("relu3", "relu2", activation_relu));
    topology.add(activation("relu4", "relu3", activation_relu));
    topology.add(activation("relu5", "relu4", activation_relu));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(engine.get_total_device_memory_size(), (uint64_t)1088);
}


TEST(memory_pool, multi_outputs_network) {
    //            -- relu -- relu1 -- relu4
    //     input<           
    //            -- relu2 --  relu3 -- relu5--relu6--relu7
    // neither of relu5, relu6 nor relu7 can share resource with relu4. 

    // uncomment this line to disable memory pool
    /*engine_configuration cfg{ false, false, false, std::string(), std::string(), true, std::string(),std::string(), 0, false };
    engine engine{ cfg };*/
    engine engine;
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 4;
    auto y_size = 4;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu", "input", activation_relu));
    topology.add(activation("relu1", "relu", activation_relu));
    topology.add(activation("relu2", "input", activation_relu));
    topology.add(activation("relu3", "relu2", activation_relu));
    topology.add(activation("relu4", "relu1", activation_relu));
    topology.add(activation("relu5", "relu3", activation_relu));
    topology.add(activation("relu6", "relu5", activation_relu));
    topology.add(activation("relu7", "relu6", activation_relu));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(engine.get_total_device_memory_size(), (uint64_t)1536);
}


TEST(memory_pool, oooq) {
    /*          -- relu1 - concat1- relu4 -- 
        input<  -- relu2 |                   >-- concat2 -- relu6
                -- relu3 --  relu5 --------- 
       neither of relu5, relu6 nor relu7 can share resource with relu4. */

    engine_configuration cfg{ false, false, false, std::string(), std::string(), true /*oooq*/, std::string(),std::string(), 0, true /*mem_pool*/ };
    engine engine{ cfg };
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 4;
    auto y_size = 4;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu1", "input", activation_relu));
    topology.add(activation("relu2", "input", activation_relu));
    topology.add(activation("relu3", "input", activation_relu));
    topology.add(concatenation("concat1", { "relu1", "relu2"},concatenation::along_f));
    topology.add(activation("relu4", "concat1", activation_relu));
    topology.add(activation("relu5", "relu3", activation_relu));
    topology.add(concatenation("concat2", { "relu4", "relu5" }, concatenation::along_f));
    topology.add(activation("relu6", "concat2", activation_relu));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(engine.get_total_device_memory_size(), (uint64_t) 5376);
}