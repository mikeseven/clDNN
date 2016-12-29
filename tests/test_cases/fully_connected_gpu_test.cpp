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
#include "api/primitives/fully_connected.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/primitives/data.hpp>

using namespace cldnn;
using namespace tests;

TEST(fully_connected_gpu, xb_f32_batch_1) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t output_x = 4, output_b = 1,  // size of whole output buffer
        input_x = 3, input_b = 1,  // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto engine = engine::create();

    auto input_prim = memory::allocate( engine, { data_types::f32, { format::xb, { input_x, input_b } } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,{ format::bx, { weight_y, weight_x } } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,{ format::x, { output_x } } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    auto topology = topology::create(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    auto network = network::build(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].id(), "full_con_prim");

    auto output_prim = outputs[0].get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.5f, output_ptr[0]);
    EXPECT_EQ(2.75f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(7.0f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_2) {
    //  Input  : 3x2
    //  Output : 4x2
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //   1       1.5  0
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //
    //  Output:
    //   2.5    2.75     0.75   7
    //   4      1        2.75   5

    const int32_t output_x = 4, output_b = 2,  // size of whole output buffer
        input_x = 3, input_b = 2,  // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto engine = engine::create();

    auto input_prim = memory::allocate(engine, { data_types::f32,{ format::xb,{ input_x, input_b } } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,{ format::bx,{ weight_y, weight_x } } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,{ format::x,{ output_x } } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    auto topology = topology::create(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    auto network = network::build(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].id(), "full_con_prim");

    auto output_prim = outputs[0].get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(4.00f, output_ptr[1]);
    EXPECT_EQ(2.75f, output_ptr[2]);
    EXPECT_EQ(1.00f, output_ptr[3]);
    EXPECT_EQ(0.75f, output_ptr[4]);
    EXPECT_EQ(2.75f, output_ptr[5]);
    EXPECT_EQ(7.00f, output_ptr[6]);
    EXPECT_EQ(5.00f, output_ptr[7]);
}

TEST(fully_connected_gpu, x_f32) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t output_x = 4,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto engine = engine::create();

    auto input_prim = memory::allocate(engine, { data_types::f32,{ format::x, { input_x } } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,{ format::bx,{ weight_y, weight_x } } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,{ format::x,{ output_x } } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    auto topology = topology::create(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    auto network = network::build(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].id(), "full_con_prim");

    auto output_prim = outputs[0].get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(2.75f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(7.00f, output_ptr[3]);
}


TEST(fully_connected_gpu, yxfn_f32) {
    //  Input  : 1x2x1x2 - 1 batch 2 feature maps of size 2x1
    //  Output : 2x1 - 2 batches 1 neuron each
    //  Weights: 2x2x1x2 - 2 neurons with weights of 2 feature maps of size 2x1
    //
    //  Input:
    //   1  -2      f0: b0
    //   3  -4      f1: b0

    //  Weights:
    //   1  -1      n0: fm0  
    //   2   0      n0: fm1
    //   3   4      n1: fm0
    //   0.5 5      n1: fm1
    //
    //  Biases:
    //   1.0 -5
    //
    //  Output:
    //   10  -28.5

    auto engine = engine::create();

    auto input_prim = memory::allocate(engine, { data_types::f32,{ format::yxfb, { 1, 2, 2, 1 } } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ 2 ,{ { 1 } }, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 2, 2, 1, 2 } } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,{ format::x,{ 2 } } });

    set_values(input_prim, { 1.f, 3.f, -2.f, -4.f });
    set_values(weights_prim, { 1.f, -1.f, 2.0f, 0.f, 3.0f, 4.0f, 0.5f, 5.0f });
    set_values(bias_prim, { 1.0f, -5.0f });

    auto topology = topology::create(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    auto network = network::build(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].id(), "full_con_prim");

    auto output_prim = outputs[0].get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(10, output_ptr[0]);
    EXPECT_EQ(-28.5, output_ptr[1]);
}

TEST(fully_connected_gpu, xb_f32_batch_1_relu) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   1.0,  -2.0,  3.0,  -4.0
    //
    //  Output:
    //   2.5   0      0.75  0

    const int32_t output_x = 4, output_b = 1,  // size of whole output buffer
        input_x = 3, input_b = 1,  // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto engine = engine::create();

    auto input_prim = memory::allocate(engine, { data_types::f32,{ format::xb,{ input_x, input_b } } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,{ format::bx,{ weight_y, weight_x } } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,{ format::x,{ output_x } } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    auto topology = topology::create(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0)
    );

    auto network = network::build(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].id(), "full_con_prim");

    auto output_prim = outputs[0].get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(0.00f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_2_relu) {
    //  Input  : 3x2
    //  Output : 4x2
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //   1       1.5  0
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //
    //  Output:
    //   2.5    0   0.75   0
    //   4      0   2.75   0

    const int32_t output_x = 4, output_b = 2,  // size of whole output buffer
        input_x = 3, input_b = 2,  // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto engine = engine::create();

    auto input_prim = memory::allocate(engine, { data_types::f32,{ format::xb,{ input_x, input_b } } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,{ format::bx,{ weight_y, weight_x } } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,{ format::x,{ output_x } } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    auto topology = topology::create(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0)
    );

    auto network = network::build(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].id(), "full_con_prim");

    auto output_prim = outputs[0].get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(4.00f, output_ptr[1]);
    EXPECT_EQ(0.00f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
    EXPECT_EQ(0.75f, output_ptr[4]);
    EXPECT_EQ(2.75f, output_ptr[5]);
    EXPECT_EQ(0.00f, output_ptr[6]);
    EXPECT_EQ(0.00f, output_ptr[7]);
}

TEST(fully_connected_gpu, x_f32_relu) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //  Output:
    //   2.5   0    0.75  0

    const int32_t output_x = 4,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto engine = engine::create();

    auto input_prim = memory::allocate(engine, { data_types::f32,{ format::x,{ input_x } } });
    //auto output_prim = memory::allocate({ memory::format::x_f32,{ 1       ,{ { output_x } }, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,{ format::bx,{ weight_y, weight_x } } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,{ format::x,{ output_x } } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    auto topology = topology::create(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0)
    );

    auto network = network::build(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].id(), "full_con_prim");

    auto output_prim = outputs[0].get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(0.00f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
}

TEST(fully_connected_gpu, x_f32_relu_with_negative_slope) {
	//  Input  : 3x1
	//  Output : 4x1
	//  Weights: 4x3
	//  Negative Slope: 0.1
	//
	//  Input:
	//  -0.5     2    0.5
	//
	//  Weights:
	//   1.5     1    0.5
	//  -1       0    0.5
	//   0.5    -0.5 -2
	//  -0.5     1    1.5
	//
	//  Biases:
	//   1.0, -2.0, 3.0, -4.0
	//  Output:
	//   2.5   -0.125    0.75  -0.1

	const int32_t output_x = 4,                 // size of whole output buffer
		input_x = 3,                 // size of whole input buffer
		weight_x = 4, weight_y = 3;  // size of whole weights buffer

    auto engine = engine::create();

    auto input_prim = memory::allocate(engine, { data_types::f32,{ format::x,{ input_x } } });
    //auto output_prim = memory::allocate({ memory::format::x_f32,{ 1       ,{ { output_x } }, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,{ format::bx,{ weight_y, weight_x } } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,{ format::x,{ output_x } } });

	set_values(input_prim, { -0.5f, 2.0f, 0.5f });
	set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
	set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    auto topology = topology::create(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0.1f)
    );

    auto network = network::build(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].id(), "full_con_prim");

    auto output_prim = outputs[0].get_memory();

    auto output_ptr = output_prim.pointer<float>();

	EXPECT_EQ(2.50f, output_ptr[0]);
	EXPECT_EQ(-0.125f, output_ptr[1]);
	EXPECT_EQ(0.75f, output_ptr[2]);
	EXPECT_EQ(-0.1f, output_ptr[3]);
}