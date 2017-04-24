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
#include "api/CPP/prior_box.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

namespace cldnn
{
	template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

template <typename T>
class prior_box_test : public ::testing::Test
{
};

typedef ::testing::Types<float, FLOAT16> prior_box_test_types;
TYPED_TEST_CASE(prior_box_test, prior_box_test_types);

TYPED_TEST(prior_box_test, test_setup_basic) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.batch[0], 1);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.feature[0], 2);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[1], 10 * 10 * 2 * 4);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[0], 1);
}

TYPED_TEST(prior_box_test, test_setup_multi_size) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4, 14 };
	std::vector<float> max_sizes = { 9, 19 };

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.batch[0], 1);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.feature[0], 2);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[1], 10 * 10 * 4 * 4);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[0], 1);
}

TYPED_TEST(prior_box_test, test_setup_no_max_size) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.batch[0], 1);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.feature[0], 2);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[1], 10 * 10 * 1 * 4);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[0], 1);
}

TYPED_TEST(prior_box_test, test_setup_multi_size_no_max_size) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4, 14 };

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.batch[0], 1);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.feature[0], 2);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[1], 10 * 10 * 2 * 4);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[0], 1);
}

TYPED_TEST(prior_box_test, test_setup_aspect_ratio_one_no_flip) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 1 , 2 };

	bool flip = false;

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios, flip));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.batch[0], 1);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.feature[0], 2);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[1], 10 * 10 * 3 * 4);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[0], 1);
}

TYPED_TEST(prior_box_test, test_setup_aspect_ratio_no_flip) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 2 , 3 };

	bool flip = false;

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios, flip));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.batch[0], 1);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.feature[0], 2);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[1], 10 * 10 * 4 * 4);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[0], 1);
}

TYPED_TEST(prior_box_test, test_setup_aspect_ratio_flip) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 2 , 3 };

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.batch[0], 1);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.feature[0], 2);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[1], 10 * 10 * 6 * 4);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[0], 1);
}

TYPED_TEST(prior_box_test, test_setup_aspect_ratio_flip_multi_size) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4, 14 };
	std::vector<float> max_sizes = { 9, 19 };
	std::vector<float> aspect_ratios = { 2 , 3 };

	bool flip = true;

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios, flip));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.batch[0], 1);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.feature[0], 2);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[1], 10 * 10 * 12 * 4);
	EXPECT_EQ(outputs.begin()->second.get_memory().get_layout().size.spatial[0], 1);
}

TYPED_TEST(prior_box_test, test_forward_basic) 
{
    engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx, { 10, 10, 10, 10} } });
	
    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };

    topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes));
    network network(engine, topology);
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "prior_box");
	
    auto output_prim = outputs.begin()->second.get_memory();
	
    auto output_ptr = output_prim.pointer<TypeParam>();
	
	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_TRUE(floating_point_equal(output_ptr[0], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[1], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[2], (TypeParam)0.07f));
	EXPECT_TRUE(floating_point_equal(output_ptr[3], (TypeParam)0.07f));
	// second prior
	EXPECT_TRUE(floating_point_equal(output_ptr[4], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[5], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[6], (TypeParam)0.08f));
	EXPECT_TRUE(floating_point_equal(output_ptr[7], (TypeParam)0.08f));
	// prior in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], (TypeParam)0.47f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], (TypeParam)0.47f));

	//check variance
	for (int d = 0; d < dim; ++d) 
	{
		EXPECT_TRUE(floating_point_equal(output_ptr[dim + d], (TypeParam)0.1f));
	}
}

TYPED_TEST(prior_box_test, test_forward_no_max_size) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<TypeParam>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_TRUE(floating_point_equal(output_ptr[0], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[1], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[2], (TypeParam)0.07f));
	EXPECT_TRUE(floating_point_equal(output_ptr[3], (TypeParam)0.07f));

	// prior in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 1 * 4 + 4 * 1 * 4], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 1 * 4 + 4 * 1 * 4 + 1], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 1 * 4 + 4 * 1 * 4 + 2], (TypeParam)0.47f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 1 * 4 + 4 * 1 * 4 + 3], (TypeParam)0.47f));

	//check variance
	for (int d = 0; d < dim; ++d) 
	{
		EXPECT_TRUE(floating_point_equal(output_ptr[dim + d], (TypeParam)0.1f));
	}
}

TYPED_TEST(prior_box_test, test_forward_variance_one) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = {};
	bool flip = true;
	bool clip = false;
	std::vector<float> variance = { 1 };

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios, flip, clip, variance));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<TypeParam>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_TRUE(floating_point_equal(output_ptr[0], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[1], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[2], (TypeParam)0.07f));
	EXPECT_TRUE(floating_point_equal(output_ptr[3], (TypeParam)0.07f));

	// second prior
	EXPECT_TRUE(floating_point_equal(output_ptr[4], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[5], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[6], (TypeParam)0.08f));
	EXPECT_TRUE(floating_point_equal(output_ptr[7], (TypeParam)0.08f));

	// prior in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], (TypeParam)0.47f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], (TypeParam)0.47f));


	//check variance
	for (int d = 0; d < dim; ++d) 
	{
		EXPECT_TRUE(floating_point_equal(output_ptr[dim + d], (TypeParam)1.f));
	}
}

TYPED_TEST(prior_box_test, test_forward_variance_multi) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = {};
	bool flip = true;
	bool clip = false;
	std::vector<float> variance = { 0.1f, 0.2f, 0.3f, 0.4f };

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios, flip, clip, variance));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<TypeParam>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_TRUE(floating_point_equal(output_ptr[0], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[1], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[2], (TypeParam)0.07f));
	EXPECT_TRUE(floating_point_equal(output_ptr[3], (TypeParam)0.07f));

	// second prior
	EXPECT_TRUE(floating_point_equal(output_ptr[4], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[5], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[6], (TypeParam)0.08f));
	EXPECT_TRUE(floating_point_equal(output_ptr[7], (TypeParam)0.08f));

	// prior in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], (TypeParam)0.47f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], (TypeParam)0.47f));

	//check variance
	for (int d = 0; d < dim; ++d) 
	{
		EXPECT_TRUE(floating_point_equal(output_ptr[dim + d], (TypeParam)(0.1f * (d % 4 + 1))));
	}
}

TYPED_TEST(prior_box_test, test_forward_aspect_ratio_no_flip) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 2 };
	bool flip = false;

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios, flip));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<TypeParam>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_TRUE(floating_point_equal(output_ptr[0], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[1], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[2], (TypeParam)0.07f));
	EXPECT_TRUE(floating_point_equal(output_ptr[3], (TypeParam)0.07f));
	// second prior
	EXPECT_TRUE(floating_point_equal(output_ptr[4], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[5], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[6], (TypeParam)0.08f));
	EXPECT_TRUE(floating_point_equal(output_ptr[7], (TypeParam)0.08f));
	// third prior
	EXPECT_TRUE(floating_point_equal(output_ptr[8], (TypeParam)(0.05f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[9], (TypeParam)(0.05f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[10], (TypeParam)(0.05f + 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[11], (TypeParam)(0.05f + 0.01f*(float)sqrt(2.f))));
	// prior in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 1], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 2], (TypeParam)0.47f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 3], (TypeParam)0.47f));
	// prior with ratio 1:2 in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 8], (TypeParam)(0.45f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 9], (TypeParam)(0.45f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 10], (TypeParam)(0.45f + 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 11], (TypeParam)(0.45f + 0.01f*(float)sqrt(2.f))));

	//check variance
	for (int d = 0; d < dim; ++d) 
	{
		EXPECT_TRUE(floating_point_equal(output_ptr[dim + d], (TypeParam)0.1f));
	}
}

TYPED_TEST(prior_box_test, test_forward_aspect_ratio) 
{
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 2 };

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<TypeParam>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_TRUE(floating_point_equal(output_ptr[0], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[1], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[2], (TypeParam)0.07f));
	EXPECT_TRUE(floating_point_equal(output_ptr[3], (TypeParam)0.07f));
	// second prior
	EXPECT_TRUE(floating_point_equal(output_ptr[4], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[5], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[6], (TypeParam)0.08f));
	EXPECT_TRUE(floating_point_equal(output_ptr[7], (TypeParam)0.08f));
	// third prior
	EXPECT_TRUE(floating_point_equal(output_ptr[8], (TypeParam)(0.05f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[9], (TypeParam)(0.05f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[10], (TypeParam)(0.05f + 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[11], (TypeParam)(0.05f + 0.01f*(float)sqrt(2.f))));
	// forth prior
	EXPECT_TRUE(floating_point_equal(output_ptr[12], (TypeParam)(0.05f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[13], (TypeParam)(0.05f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[14], (TypeParam)(0.05f + 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[15], (TypeParam)(0.05f + 0.02f*(float)sqrt(2.f))));
	// prior in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 1], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 2], (TypeParam)0.47f));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 3], (TypeParam)0.47f));
	// prior with ratio 1:2 in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 8], (TypeParam)(0.45f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 9], (TypeParam)(0.45f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 10], (TypeParam)(0.45f + 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 11], (TypeParam)(0.45f + 0.01f*(float)sqrt(2.f))));
	// prior with ratio 2:1 in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 12], (TypeParam)(0.45f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 13], (TypeParam)(0.45f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 14], (TypeParam)(0.45f + 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 15], (TypeParam)(0.45f + 0.02f*(float)sqrt(2.f))));

	//check variance
	for (int d = 0; d < dim; ++d) 
	{
		EXPECT_TRUE(floating_point_equal(output_ptr[dim + d], (TypeParam)0.1f));
	}
}

TYPED_TEST(prior_box_test, test_forward_aspect_ratio_multi_size) {
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4, 8 };
	std::vector<float> max_sizes = { 9, 18 };
	std::vector<float> aspect_ratios = { 2 };
	bool flip = true;
	bool clip = true;

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios, flip, clip));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<TypeParam>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_TRUE(floating_point_equal(output_ptr[0], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[1], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[2], (TypeParam)0.07f));
	EXPECT_TRUE(floating_point_equal(output_ptr[3], (TypeParam)0.07f));
	// second prior
	EXPECT_TRUE(floating_point_equal(output_ptr[4], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[5], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[6], (TypeParam)0.08f));
	EXPECT_TRUE(floating_point_equal(output_ptr[7], (TypeParam)0.08f));
	// third prior
	EXPECT_TRUE(floating_point_equal(output_ptr[8], (TypeParam)(0.05f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[9], (TypeParam)(0.05f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[10], (TypeParam)(0.05f + 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[11], (TypeParam)(0.05f + 0.01f*(float)sqrt(2.f))));
	// forth prior
	EXPECT_TRUE(floating_point_equal(output_ptr[12], (TypeParam)(0.05f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[13], (TypeParam)(0.05f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[14], (TypeParam)(0.05f + 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[15], (TypeParam)(0.05f + 0.02f*(float)sqrt(2.f))));
	// fifth prior
	EXPECT_TRUE(floating_point_equal(output_ptr[16], (TypeParam)0.01f));
	EXPECT_TRUE(floating_point_equal(output_ptr[17], (TypeParam)0.01f));
	EXPECT_TRUE(floating_point_equal(output_ptr[18], (TypeParam)0.09f));
	EXPECT_TRUE(floating_point_equal(output_ptr[19], (TypeParam)0.09f));
	// sixth prior
	EXPECT_TRUE(floating_point_equal(output_ptr[20], (TypeParam)0.00f));
	EXPECT_TRUE(floating_point_equal(output_ptr[21], (TypeParam)0.00f));
	EXPECT_TRUE(floating_point_equal(output_ptr[22], (TypeParam)0.11f));
	EXPECT_TRUE(floating_point_equal(output_ptr[23], (TypeParam)0.11f));
	// seventh prior
	EXPECT_TRUE(floating_point_equal(output_ptr[24], (TypeParam)0.00f));
	EXPECT_TRUE(floating_point_equal(output_ptr[25], (TypeParam)(0.05f - 0.04f / (float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[26], (TypeParam)(0.05f + 0.04f * (float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[27], (TypeParam)(0.05f + 0.04f / (float)sqrt(2.f))));
	// forth prior
	EXPECT_TRUE(floating_point_equal(output_ptr[28], (TypeParam)(0.05f - 0.04f / (float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[29], (TypeParam)0.00f));
	EXPECT_TRUE(floating_point_equal(output_ptr[30], (TypeParam)(0.05f + 0.04f / (float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[31], (TypeParam)(0.05f + 0.04f * (float)sqrt(2.f))));
	// prior in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 1], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 2], (TypeParam)0.47f));
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 3], (TypeParam)0.47f));
	// prior with ratio 1:2 in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 8], (TypeParam)(0.45f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 9], (TypeParam)(0.45f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 10], (TypeParam)(0.45f + 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 11], (TypeParam)(0.45f + 0.01f*(float)sqrt(2.f))));
	// prior with ratio 2:1 in the 5-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 12], (TypeParam)(0.45f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 13], (TypeParam)(0.45f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 14], (TypeParam)(0.45f + 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 15], (TypeParam)(0.45f + 0.02f*(float)sqrt(2.f))));

	//check variance
	for (int d = 0; d < dim; ++d) 
	{
		EXPECT_TRUE(floating_point_equal(output_ptr[dim + d], (TypeParam)0.1f));
	}
}

TYPED_TEST(prior_box_test, test_forward_fix_step) {
	engine engine;

	auto input_prim = memory::allocate(engine, { type_to_data_type<TypeParam>::value, { format::bfyx,{ 10, 10, 20, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 2 };
	std::vector<float> variance = { };
	bool flip = true;
	bool clip = false;
	float step_width = 10.f;
	float step_height = 10.f;

	topology.add(prior_box("prior_box", "input_prim", { format::bfyx,{ 1,1,100,100 } }, min_sizes, max_sizes, aspect_ratios, flip, clip, variance, step_width, step_height));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<TypeParam>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_TRUE(floating_point_equal(output_ptr[0], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[1], (TypeParam)0.03f));
	EXPECT_TRUE(floating_point_equal(output_ptr[2], (TypeParam)0.07f));
	EXPECT_TRUE(floating_point_equal(output_ptr[3], (TypeParam)0.07f));
	// second prior
	EXPECT_TRUE(floating_point_equal(output_ptr[4], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[5], (TypeParam)0.02f));
	EXPECT_TRUE(floating_point_equal(output_ptr[6], (TypeParam)0.08f));
	EXPECT_TRUE(floating_point_equal(output_ptr[7], (TypeParam)0.08f));
	// third prior
	EXPECT_TRUE(floating_point_equal(output_ptr[8], (TypeParam)(0.05f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[9], (TypeParam)(0.05f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[10], (TypeParam)(0.05f + 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[11], (TypeParam)(0.05f + 0.01f*(float)sqrt(2.f))));
	// forth prior
	EXPECT_TRUE(floating_point_equal(output_ptr[12], (TypeParam)(0.05f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[13], (TypeParam)(0.05f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[14], (TypeParam)(0.05f + 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[15], (TypeParam)(0.05f + 0.02f*(float)sqrt(2.f))));
	// prior in the 15-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4], (TypeParam)0.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 1], (TypeParam)1.43f));
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 2], (TypeParam)0.47f));
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 3], (TypeParam)1.47f));
	// prior with ratio 1:2 in the 15-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 8], (TypeParam)(0.45f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 9], (TypeParam)(1.45f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 10], (TypeParam)(0.45f + 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 11], (TypeParam)(1.45f + 0.01f*(float)sqrt(2.f))));
	// prior with ratio 2:1 in the 15-th row and 5-th col
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 12], (TypeParam)(0.45f - 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 13], (TypeParam)(1.45f - 0.02f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 14], (TypeParam)(0.45f + 0.01f*(float)sqrt(2.f))));
	EXPECT_TRUE(floating_point_equal(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 15], (TypeParam)(1.45f + 0.02f*(float)sqrt(2.f))));

	//check variance
	for (int d = 0; d < dim; ++d) 
	{
		EXPECT_TRUE(floating_point_equal(output_ptr[dim + d], (TypeParam)0.1f));
	}
}