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
#include "api/primitives/prior_box.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(prior_box, test_setup_basic) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes));
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

TEST(prior_box, test_setup_multi_size) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4, 14 };
	std::vector<float> max_sizes = { 9, 19 };

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes));
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

TEST(prior_box, test_setup_no_max_size) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes));
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

TEST(prior_box, test_setup_multi_size_no_max_size) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4, 14 };

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes));
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

TEST(prior_box, test_setup_aspect_ratio_one_no_flip) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 1 , 2 };

	bool flip = false;

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios, flip));
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

TEST(prior_box, test_setup_aspect_ratio_no_flip) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 2 , 3 };

	bool flip = false;

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios, flip));
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

TEST(prior_box, test_setup_aspect_ratio_flip) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 2 , 3 };

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios));
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

TEST(prior_box, test_setup_aspect_ratio_flip_multi_size) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4, 14 };
	std::vector<float> max_sizes = { 9, 19 };
	std::vector<float> aspect_ratios = { 2 , 3 };

	bool flip = true;

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios, flip));
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

TEST(prior_box, test_forward_basic) {
    engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32, { format::bfyx, { 10, 10, 10, 10} } });
	
    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };

    topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes));
    network network(engine, topology);
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "prior_box");
	
    auto output_prim = outputs.begin()->second.get_memory();
	
    auto output_ptr = output_prim.pointer<float>();
	
	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	const float eps = (float)1e-6;
	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_NEAR(output_ptr[0], 0.03, eps);
	EXPECT_NEAR(output_ptr[1], 0.03, eps);
	EXPECT_NEAR(output_ptr[2], 0.07, eps);
	EXPECT_NEAR(output_ptr[3], 0.07, eps);
	// second prior
	EXPECT_NEAR(output_ptr[4], 0.02, eps);
	EXPECT_NEAR(output_ptr[5], 0.02, eps);
	EXPECT_NEAR(output_ptr[6], 0.08, eps);
	EXPECT_NEAR(output_ptr[7], 0.08, eps);
	// prior in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], 0.47, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], 0.47, eps);

	//check variance
	for (int d = 0; d < dim; ++d) {
		EXPECT_NEAR(output_ptr[dim + d], 0.1, eps);
	}
}

TEST(prior_box, test_forward_no_max_size) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<float>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	const float eps = (float)1e-6;
	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_NEAR(output_ptr[0], 0.03, eps);
	EXPECT_NEAR(output_ptr[1], 0.03, eps);
	EXPECT_NEAR(output_ptr[2], 0.07, eps);
	EXPECT_NEAR(output_ptr[3], 0.07, eps);

	// prior in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[4 * 10 * 1 * 4 + 4 * 1 * 4], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 1 * 4 + 4 * 1 * 4 + 1], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 1 * 4 + 4 * 1 * 4 + 2], 0.47, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 1 * 4 + 4 * 1 * 4 + 3], 0.47, eps);

	//check variance
	for (int d = 0; d < dim; ++d) {
		EXPECT_NEAR(output_ptr[dim + d], 0.1, eps);
	}
}

TEST(prior_box, test_forward_variance_one) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = {};
	bool flip = true;
	bool clip = false;
	std::vector<float> variance = { 1 };

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios, flip, clip, variance));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<float>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	const float eps = (float)1e-6;
	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_NEAR(output_ptr[0], 0.03, eps);
	EXPECT_NEAR(output_ptr[1], 0.03, eps);
	EXPECT_NEAR(output_ptr[2], 0.07, eps);
	EXPECT_NEAR(output_ptr[3], 0.07, eps);

	// second prior
	EXPECT_NEAR(output_ptr[4], 0.02, eps);
	EXPECT_NEAR(output_ptr[5], 0.02, eps);
	EXPECT_NEAR(output_ptr[6], 0.08, eps);
	EXPECT_NEAR(output_ptr[7], 0.08, eps);

	// prior in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], 0.47, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], 0.47, eps);


	//check variance
	for (int d = 0; d < dim; ++d) {
		EXPECT_NEAR(output_ptr[dim + d], 1, eps);
	}
}

TEST(prior_box, test_forward_variance_multi) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = {};
	bool flip = true;
	bool clip = false;
	std::vector<float> variance = { 0.1f, 0.2f, 0.3f, 0.4f };

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios, flip, clip, variance));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<float>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	const float eps = (float)1e-6;
	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_NEAR(output_ptr[0], 0.03, eps);
	EXPECT_NEAR(output_ptr[1], 0.03, eps);
	EXPECT_NEAR(output_ptr[2], 0.07, eps);
	EXPECT_NEAR(output_ptr[3], 0.07, eps);

	// second prior
	EXPECT_NEAR(output_ptr[4], 0.02, eps);
	EXPECT_NEAR(output_ptr[5], 0.02, eps);
	EXPECT_NEAR(output_ptr[6], 0.08, eps);
	EXPECT_NEAR(output_ptr[7], 0.08, eps);

	// prior in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], 0.47, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], 0.47, eps);

	//check variance
	for (int d = 0; d < dim; ++d) {
		EXPECT_NEAR(output_ptr[dim + d], 0.1 * (d % 4 + 1), eps);
	}
}

TEST(prior_box, test_forward_aspect_ratio_no_flip) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 2 };
	bool flip = false;

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios, flip));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<float>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	const float eps = (float)1e-6;
	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_NEAR(output_ptr[0], 0.03, eps);
	EXPECT_NEAR(output_ptr[1], 0.03, eps);
	EXPECT_NEAR(output_ptr[2], 0.07, eps);
	EXPECT_NEAR(output_ptr[3], 0.07, eps);
	// second prior
	EXPECT_NEAR(output_ptr[4], 0.02, eps);
	EXPECT_NEAR(output_ptr[5], 0.02, eps);
	EXPECT_NEAR(output_ptr[6], 0.08, eps);
	EXPECT_NEAR(output_ptr[7], 0.08, eps);
	// third prior
	EXPECT_NEAR(output_ptr[8], 0.05 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[9], 0.05 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[10], 0.05 + 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[11], 0.05 + 0.01*sqrt(2.), eps);
	// prior in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 1], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 2], 0.47, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 3], 0.47, eps);
	// prior with ratio 1:2 in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 8], 0.45 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 9], 0.45 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 10], 0.45 + 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 3 * 4 + 4 * 3 * 4 + 11], 0.45 + 0.01*sqrt(2.), eps);

	//check variance
	for (int d = 0; d < dim; ++d) {
		EXPECT_NEAR(output_ptr[dim + d], 0.1, eps);
	}
}

TEST(prior_box, test_forward_aspect_ratio) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4 };
	std::vector<float> max_sizes = { 9 };
	std::vector<float> aspect_ratios = { 2 };

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<float>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	const float eps = (float)1e-6;
	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_NEAR(output_ptr[0], 0.03, eps);
	EXPECT_NEAR(output_ptr[1], 0.03, eps);
	EXPECT_NEAR(output_ptr[2], 0.07, eps);
	EXPECT_NEAR(output_ptr[3], 0.07, eps);
	// second prior
	EXPECT_NEAR(output_ptr[4], 0.02, eps);
	EXPECT_NEAR(output_ptr[5], 0.02, eps);
	EXPECT_NEAR(output_ptr[6], 0.08, eps);
	EXPECT_NEAR(output_ptr[7], 0.08, eps);
	// third prior
	EXPECT_NEAR(output_ptr[8], 0.05 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[9], 0.05 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[10], 0.05 + 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[11], 0.05 + 0.01*sqrt(2.), eps);
	// forth prior
	EXPECT_NEAR(output_ptr[12], 0.05 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[13], 0.05 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[14], 0.05 + 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[15], 0.05 + 0.02*sqrt(2.), eps);
	// prior in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 1], 0.43, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 2], 0.47, eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 3], 0.47, eps);
	// prior with ratio 1:2 in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 8], 0.45 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 9], 0.45 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 10], 0.45 + 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 11], 0.45 + 0.01*sqrt(2.), eps);
	// prior with ratio 2:1 in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 12], 0.45 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 13], 0.45 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 14], 0.45 + 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[4 * 10 * 4 * 4 + 4 * 4 * 4 + 15], 0.45 + 0.02*sqrt(2.), eps);

	//check variance
	for (int d = 0; d < dim; ++d) {
		EXPECT_NEAR(output_ptr[dim + d], 0.1, eps);
	}
}

TEST(prior_box, test_forward_aspect_ratio_multi_size) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 10, 10 } } });

	topology topology;
	topology.add(input_layout("input_prim", input_prim.get_layout()));

	std::vector<float> min_sizes = { 4, 8 };
	std::vector<float> max_sizes = { 9, 18 };
	std::vector<float> aspect_ratios = { 2 };
	bool flip = true;
	bool clip = true;

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios, flip, clip));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<float>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	const float eps = (float)1e-6;
	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_NEAR(output_ptr[0], 0.03, eps);
	EXPECT_NEAR(output_ptr[1], 0.03, eps);
	EXPECT_NEAR(output_ptr[2], 0.07, eps);
	EXPECT_NEAR(output_ptr[3], 0.07, eps);
	// second prior
	EXPECT_NEAR(output_ptr[4], 0.02, eps);
	EXPECT_NEAR(output_ptr[5], 0.02, eps);
	EXPECT_NEAR(output_ptr[6], 0.08, eps);
	EXPECT_NEAR(output_ptr[7], 0.08, eps);
	// third prior
	EXPECT_NEAR(output_ptr[8], 0.05 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[9], 0.05 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[10], 0.05 + 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[11], 0.05 + 0.01*sqrt(2.), eps);
	// forth prior
	EXPECT_NEAR(output_ptr[12], 0.05 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[13], 0.05 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[14], 0.05 + 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[15], 0.05 + 0.02*sqrt(2.), eps);
	// fifth prior
	EXPECT_NEAR(output_ptr[16], 0.01, eps);
	EXPECT_NEAR(output_ptr[17], 0.01, eps);
	EXPECT_NEAR(output_ptr[18], 0.09, eps);
	EXPECT_NEAR(output_ptr[19], 0.09, eps);
	// sixth prior
	EXPECT_NEAR(output_ptr[20], 0.00, eps);
	EXPECT_NEAR(output_ptr[21], 0.00, eps);
	EXPECT_NEAR(output_ptr[22], 0.11, eps);
	EXPECT_NEAR(output_ptr[23], 0.11, eps);
	// seventh prior
	EXPECT_NEAR(output_ptr[24], 0.00, eps);
	EXPECT_NEAR(output_ptr[25], 0.05 - 0.04 / sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[26], 0.05 + 0.04*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[27], 0.05 + 0.04 / sqrt(2.), eps);
	// forth prior
	EXPECT_NEAR(output_ptr[28], 0.05 - 0.04 / sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[29], 0.00, eps);
	EXPECT_NEAR(output_ptr[30], 0.05 + 0.04 / sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[31], 0.05 + 0.04*sqrt(2.), eps);
	// prior in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4], 0.43, eps);
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 1], 0.43, eps);
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 2], 0.47, eps);
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 3], 0.47, eps);
	// prior with ratio 1:2 in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 8], 0.45 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 9], 0.45 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 10], 0.45 + 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 11], 0.45 + 0.01*sqrt(2.), eps);
	// prior with ratio 2:1 in the 5-th row and 5-th col
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 12], 0.45 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 13], 0.45 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 14], 0.45 + 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[8 * 10 * 4 * 4 + 8 * 4 * 4 + 15], 0.45 + 0.02*sqrt(2.), eps);

	//check variance
	for (int d = 0; d < dim; ++d) {
		EXPECT_NEAR(output_ptr[dim + d], 0.1, eps);
	}
}

TEST(prior_box, test_forward_fix_step) {
	engine engine;

	auto input_prim = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 10, 10, 20, 10 } } });

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

	topology.add(prior_box("prior_box", "input_prim", { format::yx,{ 100,100 } }, min_sizes, max_sizes, aspect_ratios, flip, clip, variance, step_width, step_height));
	network network(engine, topology);
	network.set_input_data("input_prim", input_prim);

	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "prior_box");

	auto output_prim = outputs.begin()->second.get_memory();

	auto output_ptr = output_prim.pointer<float>();

	int dim = output_prim.get_layout().size.spatial[0] * output_prim.get_layout().size.spatial[1];

	const float eps = (float)1e-6;
	// pick a few generated priors and compare against the expected number.
	// first prior
	EXPECT_NEAR(output_ptr[0], 0.03, eps);
	EXPECT_NEAR(output_ptr[1], 0.03, eps);
	EXPECT_NEAR(output_ptr[2], 0.07, eps);
	EXPECT_NEAR(output_ptr[3], 0.07, eps);
	// second prior
	EXPECT_NEAR(output_ptr[4], 0.02, eps);
	EXPECT_NEAR(output_ptr[5], 0.02, eps);
	EXPECT_NEAR(output_ptr[6], 0.08, eps);
	EXPECT_NEAR(output_ptr[7], 0.08, eps);
	// third prior
	EXPECT_NEAR(output_ptr[8], 0.05 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[9], 0.05 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[10], 0.05 + 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[11], 0.05 + 0.01*sqrt(2.), eps);
	// forth prior
	EXPECT_NEAR(output_ptr[12], 0.05 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[13], 0.05 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[14], 0.05 + 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[15], 0.05 + 0.02*sqrt(2.), eps);
	// prior in the 15-th row and 5-th col
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4], 0.43, eps);
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 1], 1.43, eps);
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 2], 0.47, eps);
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 3], 1.47, eps);
	// prior with ratio 1:2 in the 15-th row and 5-th col
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 8], 0.45 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 9], 1.45 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 10], 0.45 + 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 11], 1.45 + 0.01*sqrt(2.), eps);
	// prior with ratio 2:1 in the 15-th row and 5-th col
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 12], 0.45 - 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 13], 1.45 - 0.02*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 14], 0.45 + 0.01*sqrt(2.), eps);
	EXPECT_NEAR(output_ptr[14 * 10 * 4 * 4 + 4 * 4 * 4 + 15], 1.45 + 0.02*sqrt(2.), eps);

	//check variance
	for (int d = 0; d < dim; ++d) {
		EXPECT_NEAR(output_ptr[dim + d], 0.1, eps);
	}
}