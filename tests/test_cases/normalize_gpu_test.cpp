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
#include <api/CPP/memory.hpp>
#include <api/CPP/input_layout.hpp>
#include "api/CPP/normalize.hpp"
#include "api/CPP/reorder.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <iostream>
#include "float16.h"

using namespace cldnn;

class normalize_gpu_test : public ::testing::TestWithParam<cldnn::format>
{
};

TEST_P(normalize_gpu_test, normalize_test_across_spatial)
{
	using namespace cldnn;
	using namespace tests;

	engine engine;

	auto input = memory::allocate(engine, { data_types::f32, GetParam(),{ 7, 10, 17, 13 } });
	auto scale = memory::allocate(engine, { data_types::f32, GetParam(),{ 1, 1, 1, 1 } });

	tests::set_random_values<float>(input, -100, 100);

	std::vector<float> scale_input_vec = { 1.f };
	set_values(scale, scale_input_vec);

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(input_layout("scale", scale.get_layout()));
	topology.add(normalize("normalize", "input", "scale"));

	network network(engine, topology);

	network.set_input_data("input", input);
	network.set_input_data("scale", scale);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[3];
	int x_size = output_sizes[2];

	for (int b = 0; b < batch_size; ++b)
	{
		float norm = 0;
		for (int f = 0; f < feature_size; ++f) 
		{
			for (int y = 0; y < y_size; ++y) 
			{
				for (int x = 0; x < x_size; ++x)
				{
					size_t data_index = generic_test::get_linear_index(output.get_layout(), b, f, y, x);
					float data = buff[data_index];
					norm += data * data;
				}
			}
		}

		EXPECT_NEAR(scale_input_vec[0], sqrt(norm), 1e-5f);
	}  
}

TEST_P(normalize_gpu_test, normalize_test_across_spatial_scale_channels_shared)
{
	using namespace cldnn;
	using namespace tests;

	engine engine;

	auto input = memory::allocate(engine, { data_types::f32, GetParam(),{ 12, 5, 7, 2 } });
	auto scale = memory::allocate(engine, { data_types::f32, GetParam(),{ 1, 1, 1, 1 } });

	tests::set_random_values<float>(input, -100, 100);

	std::vector<float> scale_input_vec = { 5.2f };
	set_values(scale, scale_input_vec);

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(input_layout("scale", scale.get_layout()));
	topology.add(normalize("normalize", "input", "scale"));

	network network(engine, topology);

	network.set_input_data("input", input);
	network.set_input_data("scale", scale);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[3];
	int x_size = output_sizes[2];

	for (int b = 0; b < batch_size; ++b)
	{
		float norm = 0;
		for (int f = 0; f < feature_size; ++f)
		{
			for (int y = 0; y < y_size; ++y)
			{
				for (int x = 0; x < x_size; ++x)
				{
					size_t data_index = generic_test::get_linear_index(output.get_layout(), b, f, y, x);
					float data = buff[data_index];
					norm += data * data;
				}
			}
		}

		EXPECT_NEAR(scale_input_vec[0], sqrt(norm), 1e-5f);
	}
}

TEST_P(normalize_gpu_test, normalize_test_across_spatial_scale_channels_not_shared)
{
	using namespace cldnn;
	using namespace tests;

	engine engine;

	auto input = memory::allocate(engine, { data_types::f32, GetParam(),{ 12, 5, 7, 2 } });
	auto scale = memory::allocate(engine, { data_types::f32, GetParam(),{ 1, 1, 5, 1 } });

	tests::set_random_values<float>(input, -100, 100);

	std::vector<float> scale_input_vec = { 3.4f, 3.4f, 3.4f, 3.4f, 3.4f };
	set_values(scale, scale_input_vec);

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(input_layout("scale", scale.get_layout()));
	topology.add(normalize("normalize", "input", "scale"));

	network network(engine, topology);

	network.set_input_data("input", input);
	network.set_input_data("scale", scale);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[3];
	int x_size = output_sizes[2];

	for (int b = 0; b < batch_size; ++b)
	{
		float norm = 0;
		for (int f = 0; f < feature_size; ++f)
		{
			for (int y = 0; y < y_size; ++y)
			{
				for (int x = 0; x < x_size; ++x)
				{
					size_t data_index = generic_test::get_linear_index(output.get_layout(), b, f, y, x);
					float data = buff[data_index];
					norm += data * data;
				}
			}
		}

		EXPECT_NEAR(scale_input_vec[0], sqrt(norm), 1e-5f);
	}
}


TEST_P(normalize_gpu_test, normalize_test_within_spatial)
{
	using namespace cldnn;
	using namespace tests;

	engine engine;

	auto input = memory::allocate(engine, { data_types::f32, GetParam(),{ 5, 7, 19, 12 } });
	auto scale = memory::allocate(engine, { data_types::f32, GetParam(),{ 1, 1, 1, 1 } });

	tests::set_random_values<float>(input, -100, 100);

	std::vector<float> scale_input_vec = { 1.f };
	set_values(scale, scale_input_vec);

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(input_layout("scale", scale.get_layout()));
	topology.add(normalize("normalize", "input", "scale", false));

	network network(engine, topology);

	network.set_input_data("input", input);
	network.set_input_data("scale", scale);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[3];
	int x_size = output_sizes[2];

	for (int b = 0; b < batch_size; ++b) 
	{
		for (int y = 0; y < y_size; ++y) 
		{
			for (int x = 0; x < x_size; ++x) 
			{
				float norm = 0;
				for (int f = 0; f < feature_size; ++f) 
				{
					size_t data_index = generic_test::get_linear_index(output.get_layout(), b, f, y, x);
					float data = buff[data_index];
					norm += data * data;
				}

				EXPECT_NEAR(scale_input_vec[0], sqrt(norm), 1e-5f);
			}
		}
	}
}

TEST_P(normalize_gpu_test, normalize_test_within_spatial_scale_channels_shared)
{
	using namespace cldnn;
	using namespace tests;

	engine engine;

	auto input = memory::allocate(engine, { data_types::f32, GetParam(),{ 9, 4, 18, 23 } });
	auto scale = memory::allocate(engine, { data_types::f32, GetParam(),{ 1, 1, 1, 1 } });

	tests::set_random_values<float>(input, -100, 100);

	std::vector<float> scale_input_vec = { 3.2f };
	set_values(scale, scale_input_vec);

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(input_layout("scale", scale.get_layout()));
	topology.add(normalize("normalize", "input", "scale", false));

	network network(engine, topology);

	network.set_input_data("input", input);
	network.set_input_data("scale", scale);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[3];
	int x_size = output_sizes[2];

	for (int b = 0; b < batch_size; ++b)
	{
		for (int y = 0; y < y_size; ++y)
		{
			for (int x = 0; x < x_size; ++x)
			{
				float norm = 0;
				for (int f = 0; f < feature_size; ++f)
				{
					size_t data_index = generic_test::get_linear_index(output.get_layout(), b, f, y, x);
					float data = buff[data_index];
					norm += data * data;
				}

				EXPECT_NEAR(scale_input_vec[0], sqrt(norm), 1e-5f);
			}
		}
	}
}

TEST_P(normalize_gpu_test, normalize_test_within_spatial_scale_channels_not_shared)
{
	using namespace cldnn;
	using namespace tests;

	engine engine;

	auto input = memory::allocate(engine, { data_types::f32, GetParam(),{ 9, 4, 18, 23 } });
	auto scale = memory::allocate(engine, { data_types::f32, GetParam(),{ 1, 1, 4, 1 } });

	tests::set_random_values<float>(input, -100, 100);

	std::vector<float> scale_input_vec = { 7.2f, 7.2f, 7.2f, 7.2f };
	set_values(scale, scale_input_vec);

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(input_layout("scale", scale.get_layout()));
	topology.add(normalize("normalize", "input", "scale", false));

	network network(engine, topology);

	network.set_input_data("input", input);
	network.set_input_data("scale", scale);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[3];
	int x_size = output_sizes[2];

	for (int b = 0; b < batch_size; ++b)
	{
		for (int y = 0; y < y_size; ++y)
		{
			for (int x = 0; x < x_size; ++x)
			{
				float norm = 0;
				for (int f = 0; f < feature_size; ++f)
				{
					size_t data_index = generic_test::get_linear_index(output.get_layout(), b, f, y, x);
					float data = buff[data_index];
					norm += data * data;
				}

				EXPECT_NEAR(scale_input_vec[0], sqrt(norm), 1e-5f);
			}
		}
	}
}

struct custom_param_name_functor_normalize_test {
	std::string operator()(const ::testing::TestParamInfo<cldnn::format>& info) {
		return std::to_string(info.index);
	}
};

INSTANTIATE_TEST_CASE_P(normalize_gpu_test,
						normalize_gpu_test,
						::testing::Values(format::bfyx, format::yxfb),
						custom_param_name_functor_normalize_test());


class normalize_test : public tests::generic_test
{

public:

	virtual void SetUp()
	{
		max_ulps_diff_allowed = 9;
	}

	static void TearDownTestCase() 
	{
		for (auto generic_params : all_generic_params_scale_shared)
		{
			delete generic_params;
		}

		for (auto generic_params : all_generic_params_scale_per_channel)
		{
			delete generic_params;
		}

		for (auto layer_params : all_layer_params)
		{
			delete layer_params;
		}
	}

	static std::vector<cldnn::primitive*> generate_specific_test_params()
	{
		// No padding
		all_layer_params.push_back(new normalize("normalize", "input0", "input1", true));
		all_layer_params.push_back(new normalize("normalize", "input0", "input1", false));
		all_layer_params.push_back(new normalize("normalize", "input0", "input1", true, 12.3f));
		all_layer_params.push_back(new normalize("normalize", "input0", "input1", false, 15.9f));
		
		// Output padding
		all_layer_params.push_back(new normalize("normalize", "input0", "input1", true, 1e-10f, { { 0, 0, 6, 13 }, 0 }));
		all_layer_params.push_back(new normalize("normalize", "input0", "input1", false, 1e-10f, { { 0, 0, 12, 4 }, 0 }));
		all_layer_params.push_back(new normalize("normalize", "input0", "input1", true, 1e-10f, { { 0, 0, 11, 5 }, { 0, 0, 3, 19 } }));
		all_layer_params.push_back(new normalize("normalize", "input0", "input1", false, 1e-10f, { { 0, 0, 5, 13 }, { 0, 0, 1, 2 } }));
		
		// Input padding (output of reorder layer)
		all_layer_params.push_back(new normalize("normalize", "reorder0", "input1", true, 1e-10f, {{ 0, 0, 0, 0 },0 }));
		all_layer_params.push_back(new normalize("normalize", "reorder0", "input1", false, 1e-10f, {{ 0, 0, 0, 0 },0 }));
		
		// Input padding (output of reorder layer) + Output padding
		all_layer_params.push_back(new normalize("normalize", "reorder0", "input1", true, 1e-10f, { { 0, 0, 2, 1 },{ 0, 0, 4, 3 } }));
		all_layer_params.push_back(new normalize("normalize", "reorder0", "input1", false, 1e-10f, { { 0, 0, 3, 4 },{ 0, 0, 1, 2 } }));

		return all_layer_params;
	}

	static std::vector<std::tuple<tests::test_params*, cldnn::primitive*>> generate_all_test_params()
	{
		generic_test::generate_generic_test_params(all_generic_params_scale_shared);
		generic_test::generate_generic_test_params(all_generic_params_scale_per_channel);
		generate_specific_test_params();

		// Prepare scale input for normalize layer (size should be 1 since the scale is shared across the channels).
		for (tests::test_params* test_param : all_generic_params_scale_shared)
		{
			test_param->input_layouts.push_back(cldnn::tensor(1, 1, 1, 1));
		}

		// Prepare scale input for normalize layer (size should be equal to input feature size - one scale per channel).
		for (tests::test_params* test_param : all_generic_params_scale_per_channel)
		{
			cldnn::tensor input_size = test_param->input_layouts[0];
			test_param->input_layouts.push_back(cldnn::tensor(1, 1, input_size.feature[0], 1));
		}

		// Create all the combinations for the test.
		for (cldnn::primitive* layer_param : all_layer_params)
		{
			for (tests::test_params* test_param : all_generic_params_scale_shared)
			{
				all_test_params.push_back(std::make_tuple(test_param, layer_param));
			}

			for (tests::test_params* test_param : all_generic_params_scale_per_channel)
			{
				all_test_params.push_back(std::make_tuple(test_param, layer_param));
			}
		}

		return all_test_params;
	}

	virtual bool is_format_supported(cldnn::format format)
	{
		return ((format == cldnn_format_type::cldnn_format_bfyx) || (format == cldnn_format_type::cldnn_format_yxfb));
	}

	virtual void prepare_input_for_test(std::vector<cldnn::memory>& inputs)
	{
		if (generic_params->data_type == data_types::f32)
		{
			prepare_input_for_test_typed<float>(inputs);
		}
		else
		{
			prepare_input_for_test_typed<FLOAT16>(inputs);
		}
	}

	template<typename Type>
	void prepare_input_for_test_typed(std::vector<cldnn::memory>& inputs)
	{
		// Update scale values.
		auto scale_input = inputs[1];
		tests::set_random_values<Type>(scale_input, -2, 2);
	}


	template<typename Type>
	memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
	{
		const cldnn::normalize* normalize = (cldnn::normalize*)layer_params;

		//Output is bfyx
        data_types dt = inputs[0].get_layout().data_type;
		auto output = memory::allocate( engine, cldnn::layout(dt, cldnn::format::bfyx, inputs[0].get_layout().size, normalize->output_padding) );

		float epsilon = normalize->epsilon;

		auto input_mem = inputs[0].pointer<Type>();
		auto scale_mem = inputs[1].pointer<Type>();
		auto output_mem = output.pointer<Type>();
		int batch = inputs[0].get_layout().size.batch[0];
		int feature = inputs[0].get_layout().size.feature[0];
		int height = inputs[0].get_layout().size.spatial[1];
		int width = inputs[0].get_layout().size.spatial[0];

		int scale_feature_size = inputs[1].get_layout().size.spatial[0];

		int output_height = output.get_layout().get_buffer_size().spatial[1];
		int output_width = output.get_layout().get_buffer_size().spatial[0];

		assert((scale_feature_size == 1) || (scale_feature_size == feature));

		//Initialized output with zeros.
        std::fill(output_mem.begin(), output_mem.end(), static_cast<Type>(0));

		if (normalize->across_spatial)
		{
			for (int n = 0; n < batch; ++n)
			{
				//Compute norm per batch
				float norm = epsilon;
				for (int c = 0; c < feature; ++c)
				{
					for (int h = 0; h < height; ++h)
					{
						for (int w = 0; w < width; ++w)
						{
							size_t input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w);
							float value = (float)input_mem[input_index];
							norm += value * value;
						}
					}
				}
				norm = pow(norm, -0.5f);
				
				//Scale the data
				for (int c = 0; c < feature; ++c)
				{
					int scale_index = (scale_feature_size == 1) ? 0 : c;
					for (int h = 0; h < height; ++h)
					{
						for (int w = 0; w < width; ++w)
						{
							size_t input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w);

							int output_index = (n * feature + c) * output_height * output_width;
							tensor lower_padding = normalize->output_padding.lower_size();
							output_index += (lower_padding.spatial[1] + h) * output_width + lower_padding.spatial[0] + w;

							output_mem[output_index] = (Type)norm * input_mem[input_index] * scale_mem[scale_index];
						}
					}
				}
			}
		}
		else
		{
			for (int n = 0; n < batch; ++n)
			{
				for (int h = 0; h < height; ++h)
				{
					for (int w = 0; w < width; ++w)
					{
						//Compute norm per (x,y)
						float norm = epsilon;
						for (int c = 0; c < feature; ++c)
						{
							size_t input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w);
							float value = (float)input_mem[input_index];
							norm += value * value;
						}
						norm = pow(norm, -0.5f);

						//Scale the data
						for (int c = 0; c < feature; ++c)
						{
							int scale_index = (scale_feature_size == 1) ? 0 : c;
							size_t input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w);

							int output_index = (n * feature + c) * output_height * output_width;
							tensor lower_padding = normalize->output_padding.lower_size();
							output_index += (lower_padding.spatial[1] + h) * output_width + lower_padding.spatial[0] + w;

							output_mem[output_index] = (Type)norm * input_mem[input_index] * scale_mem[scale_index];
						}
					}
				}
			}
		}

		return output;
	}

	virtual memory generate_reference(const std::vector<cldnn::memory>& inputs)
	{
		if (generic_params->data_type == data_types::f32)
		{
			return generate_reference_typed<float>(inputs);
		}
		else
		{
			return generate_reference_typed<FLOAT16>(inputs);
		}		
	}

private:

	static std::vector<tests::test_params*> all_generic_params_scale_shared;
	static std::vector<tests::test_params*> all_generic_params_scale_per_channel;
	static std::vector<cldnn::primitive*> all_layer_params;
	static std::vector<std::tuple<tests::test_params*, cldnn::primitive*>> all_test_params;
	
};

std::vector<tests::test_params*> normalize_test::all_generic_params_scale_shared = {};
std::vector<tests::test_params*> normalize_test::all_generic_params_scale_per_channel = {};
std::vector<cldnn::primitive*> normalize_test::all_layer_params = {};
std::vector<std::tuple<tests::test_params*, cldnn::primitive*>> normalize_test::all_test_params = {};

TEST_P(normalize_test, DISABLED_test_all)
{
	run_single_test();
}

INSTANTIATE_TEST_CASE_P(NORMALIZE, 
						normalize_test, 
						::testing::ValuesIn(normalize_test::generate_all_test_params()),
						tests::generic_test::custom_param_name_functor());
