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
#include "api/primitives/normalize.hpp"
#include "api/primitives/reorder.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
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

	auto input = memory::allocate(engine, { data_types::f32,{ GetParam(),{ 7, 10, 13, 17 } } });

	tests::set_random_values<float>(input, -100, 100);

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(normalize("normalize", "input"));

	network network(engine, topology);

	network.set_input_data("input", input);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.transform(format::bfyx, 0).sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[2];
	int x_size = output_sizes[3];

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

		EXPECT_NEAR(1, sqrt(norm), 1e-5f);
	}  
}

TEST_P(normalize_gpu_test, normalize_test_across_spatial_scale)
{
	using namespace cldnn;
	using namespace tests;

	engine engine;

	auto input = memory::allocate(engine, { data_types::f32,{ GetParam(),{ 12, 5, 2, 7 } } });

	tests::set_random_values<float>(input, -100, 100);

	float scale = 5.2f;

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(normalize("normalize", "input", true, 1e-10f, scale));

	network network(engine, topology);

	network.set_input_data("input", input);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.transform(format::bfyx, 0).sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[2];
	int x_size = output_sizes[3];

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

		EXPECT_NEAR(scale, sqrt(norm), 1e-5f);
	}
}

TEST_P(normalize_gpu_test, normalize_test_within_spatial)
{
	using namespace cldnn;
	using namespace tests;

	engine engine;

	auto input = memory::allocate(engine, { data_types::f32,{ GetParam(),{ 5, 7, 12, 19 } } });

	tests::set_random_values<float>(input, -100, 100);

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(normalize("normalize", "input", false));

	network network(engine, topology);

	network.set_input_data("input", input);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.transform(format::bfyx, 0).sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[2];
	int x_size = output_sizes[3];

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

				EXPECT_NEAR(1, sqrt(norm), 1e-5f);
			}
		}
	}
}

TEST_P(normalize_gpu_test, normalize_test_within_spatial_scale)
{
	using namespace cldnn;
	using namespace tests;

	engine engine;

	auto input = memory::allocate(engine, { data_types::f32,{ GetParam(),{ 9, 4, 23, 18 } } });

	tests::set_random_values<float>(input, -100, 100);

	float scale = 3.2f;

	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(normalize("normalize", "input", false, 1e-10f, scale));

	network network(engine, topology);

	network.set_input_data("input", input);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "normalize");

	auto output = outputs.begin()->second.get_memory();

	auto buff = output.pointer<float>();

	auto output_sizes = output.get_layout().size.transform(format::bfyx, 0).sizes();

	int batch_size = output_sizes[0];
	int feature_size = output_sizes[1];
	int y_size = output_sizes[2];
	int x_size = output_sizes[3];

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

				EXPECT_NEAR(scale, sqrt(norm), 1e-5f);
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
		for (auto generic_params : all_generic_params)
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
		all_layer_params.push_back(new normalize("normalize", "input0", true));
		all_layer_params.push_back(new normalize("normalize", "input0", false));
		all_layer_params.push_back(new normalize("normalize", "input0", true, 12.3f));
		all_layer_params.push_back(new normalize("normalize", "input0", false, 15.9f));
		all_layer_params.push_back(new normalize("normalize", "input0", true, 1e-10f, 2.2f));
		all_layer_params.push_back(new normalize("normalize", "input0", false, 1e-10f, -2.4f));
		
		// Output padding
		all_layer_params.push_back(new normalize("normalize", "input0", true, 1e-10f, 1.f, { format::yx,{ 0, 0 } }, { format::yx,{ 13, 6 } }));
		all_layer_params.push_back(new normalize("normalize", "input0", false, 1e-10f, 2.f, { format::yx,{ 0, 0 } }, { format::yx,{ 4, 12 } }));
		all_layer_params.push_back(new normalize("normalize", "input0", true, 1e-10f, 1.2f, { format::yx,{ 0, 0 } }, { format::yx,{ 5, 11 },{ 19, 3 } }));
		all_layer_params.push_back(new normalize("normalize", "input0", false, 1e-10f, -1.f, { format::yx,{ 0, 0 } }, { format::yx,{ 13, 5 },{ 2, 1 } }));
		
		// Input padding (output of reorder layer)
		all_layer_params.push_back(new normalize("normalize", "reorder0", true, 1e-10f, 0.5f, { format::yx,{ 0, 0 } }, { format::yx,{ 0, 0 } }));
		all_layer_params.push_back(new normalize("normalize", "reorder0", false, 1e-10f, -0.5f, { format::yx,{ 0, 0 } }, { format::yx,{ 0, 0 } }));
		
		// Input padding (output of reorder layer) + Output padding
		all_layer_params.push_back(new normalize("normalize", "reorder0", true, 1e-10f, 1.5f, { format::yx,{ 0, 0 } }, { format::yx,{ 1, 2 },{ 3, 4 } }));
		all_layer_params.push_back(new normalize("normalize", "reorder0", false, 1e-10f, 1.8f, { format::yx,{ 0, 0 } }, { format::yx,{ 4, 3 },{ 2, 1 } }));

		return all_layer_params;
	}

	static std::vector<tests::test_params*> generate_generic_test_params()
	{
		return generic_test::generate_generic_test_params(all_generic_params);
	}

	virtual bool is_format_supported(cldnn::format format)
	{
		return ((format == cldnn_format_type::cldnn_format_bfyx) || (format == cldnn_format_type::cldnn_format_yxfb));
	}

	template<typename Type>
	memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
	{
		const cldnn::normalize* normalize = (cldnn::normalize*)layer_params;

		//Output is bfyx
        data_types dt = inputs[0].get_layout().data_type;
		auto output = memory::allocate( engine, cldnn::layout(dt, inputs[0].get_layout().size.add(normalize->output_padding.lower_size()).add(normalize->output_padding.upper_size()).transform(cldnn::format::bfyx, 0)) );

		assert(!normalize->input_padding);

		float epsilon = normalize->epsilon;
		Type scale = normalize->scale_factor;

		auto input_mem = inputs[0].pointer<Type>();
		auto output_mem = output.pointer<Type>();
		int batch = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0];
		int feature = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1];
		int height = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2];
		int width = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3];

		int output_height = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2];
		int output_width = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3];

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
					for (int h = 0; h < height; ++h)
					{
						for (int w = 0; w < width; ++w)
						{
							size_t input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w);

							int output_index = (n * feature + c) * output_height * output_width;
							tensor lower_padding = normalize->output_padding.lower_size().transform(cldnn::format::bfyx, 0);
							output_index += (lower_padding.sizes()[2] + h) * output_width + lower_padding.sizes()[3] + w;

							output_mem[output_index] = (Type)norm * input_mem[input_index] * scale;
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
							size_t input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w);

							int output_index = (n * feature + c) * output_height * output_width;
							tensor lower_padding = normalize->output_padding.lower_size().transform(cldnn::format::bfyx, 0);
							output_index += (lower_padding.sizes()[2] + h) * output_width + lower_padding.sizes()[3] + w;

							output_mem[output_index] = (Type)norm * input_mem[input_index] * scale;
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

	static std::vector<tests::test_params*> all_generic_params;
	static std::vector<cldnn::primitive*> all_layer_params;
	
};

std::vector<cldnn::primitive*> normalize_test::all_layer_params = {};
std::vector<tests::test_params*> normalize_test::all_generic_params = {};

TEST_P(normalize_test, DISABLED_test_all)
{
	run_single_test();
}

INSTANTIATE_TEST_CASE_P(NORMALIZE, 
						normalize_test, 
						::testing::Combine(::testing::ValuesIn(normalize_test::generate_generic_test_params()),
										   ::testing::ValuesIn(normalize_test::generate_specific_test_params())), 
						tests::generic_test::custom_param_name_functor());

