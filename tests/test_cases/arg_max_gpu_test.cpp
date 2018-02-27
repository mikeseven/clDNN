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

#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/arg_max_min.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace std;
using namespace tests;


TEST(arg_max_gpu, base) {
	//  Input  : 2x3x2x2
	static const int32_t x_size = 2, y_size = 2, feature_num = 3,
		batch_num = 2;
	engine engine;

	auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(arg_max_min("arg_max", "input", arg_max_min::max));

	vector<float> input_vec = {
			//y0x0 y0x1 y1x0 y1x1
		/*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
		/*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
		/*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

		/*b1f0*/3.f,  0.5f,  7.f,   10.f,
		/*b1f1*/4.f,  0.5f,  8.f,   8.2f,
		/*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
	};
	set_values(input, input_vec);

	network network(engine, topology);

	network.set_input_data("input", input);
	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "arg_max");

	auto output = outputs.at("arg_max").get_memory();
	auto output_ptr = output.pointer<float>();;
	float out_buffer[batch_num];
	for (uint32_t i = 0; i < batch_num; i++)
	{
		out_buffer[i] = get_value<float>(output_ptr, i);
	}	
	int size = x_size * y_size * feature_num;
	float index;
	float value;
	for (int i = 0; i < batch_num; i++) {
		EXPECT_GE(out_buffer[i], 0);
		EXPECT_LT(out_buffer[i], size);
		index = out_buffer[i];
		value = input_vec[i*size + (int)index];
		for (int j = 0; j < size; j++)
		{
			EXPECT_LE(input_vec[i*size + j], value);
		}
	}
}

TEST(arg_min_gpu, base) {
	//  Input  : 2x3x2x2
	static const int32_t x_size = 2, y_size = 2, feature_num = 3,
		batch_num = 2;
	engine engine;

	auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(arg_max_min("arg_max", "input", arg_max_min::min));

	vector<float> input_vec = {
		//y0x0 y0x1 y1x0 y1x1
		/*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
		/*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
		/*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

		/*b1f0*/3.f,  0.5f,  7.f,   10.f,
		/*b1f1*/4.f,  0.5f,  8.f,   8.2f,
		/*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
	};
	set_values(input, input_vec);

	network network(engine, topology);

	network.set_input_data("input", input);
	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "arg_max");

	auto output = outputs.at("arg_max").get_memory();
	auto output_ptr = output.pointer<float>();;
	float out_buffer[batch_num];
	for (uint32_t i = 0; i < batch_num; i++)
	{
		out_buffer[i] = get_value<float>(output_ptr, i);
	}
	int size = x_size * y_size * feature_num;
	float index;
	float value;
	for (int i = 0; i < batch_num; i++) {
		EXPECT_GE(out_buffer[i], 0);
		EXPECT_LT(out_buffer[i], size);
		index = out_buffer[i];
		value = input_vec[i*size + (int)index];
		for (int j = 0; j < size; j++)
		{
			EXPECT_GE(input_vec[i*size + j], value);
		}
	}
}


class arg_max_test : public tests::generic_test
{

public:
	arg_max_test() : tests::generic_test()
	{
	}

	virtual void SetUp() override
	{
		max_ulps_diff_allowed = 6;
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
		all_layer_params.push_back(new arg_max_min("arg_max_min", "input0", arg_max_min::max));

		//The test checks only valid combinations.
		//TODO: add more combinations.

		return all_layer_params;
	}

	static std::vector<tests::test_params*> generate_generic_test_params()
	{
		return generic_test::generate_generic_test_params(all_generic_params);
	}

	virtual bool is_format_supported(cldnn::format format) override
	{
		return
			format == cldnn_format_type::cldnn_format_yxfb ||
			format == cldnn_format_type::cldnn_format_bfyx ||
			format == cldnn_format_type::cldnn_format_byxf;
	}

	template<typename Type>
	memory generate_reference_typed(const std::vector<memory> & inputs)
	{
		assert(inputs.size() == 1);
		const memory & input = inputs[0];

		//Output is bfyx
		auto output = memory::allocate(engine, cldnn::layout(input.get_layout().data_type, input.get_layout().format, input.get_layout().size));

		//        const auto params = static_cast<cldnn::arg_max *>(layer_parmas);

		const auto in0_mem = input.pointer<Type>();
		auto out_mem = output.pointer<Type>();

		const int in0_b = input.get_layout().size.sizes()[0];
		const int in0_f = input.get_layout().size.sizes()[1];
		const int in0_h = input.get_layout().size.sizes()[3];
		const int in0_w = input.get_layout().size.sizes()[2];

		//        const int out_b = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0];
		//        const int out_f = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1];
		//        const int out_h = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2];
		//        const int out_w = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3];

		//        assert(in0_b == out_b);
		//        assert(in0_f == out_f);
		//        assert(in0_h == out_h);
		//        assert(in0_w == out_w);

		std::vector<float> cached_exp_vals;
		cached_exp_vals.resize(in0_f);

		const auto input_desc = get_linear_memory_desc(input.get_layout());

		for (int n = 0; n < in0_b; ++n)
			for (int y = 0; y < in0_h; ++y)
				for (int x = 0; x < in0_w; ++x)
				{
					float max_val = -std::numeric_limits<float>::infinity();

					for (int c = 0; c < in0_f; ++c)
					{
						const size_t in0_idx = get_linear_index(input.get_layout(), n, c, y, x, input_desc);

						max_val = std::max(max_val, static_cast<float>(in0_mem[in0_idx]));
					}

					float Z = 0;

					for (int c = 0; c < in0_f; ++c)
					{
						const size_t in0_idx = get_linear_index(input.get_layout(), n, c, y, x, input_desc);

						float tmp = static_cast<float>((Type)std::exp(static_cast<float>(in0_mem[in0_idx]) - max_val));
						Z += tmp;
						cached_exp_vals[c] = tmp;
					}

					for (int c = 0; c < in0_f; ++c)
					{
						const size_t out_idx = get_linear_index(output.get_layout(), n, c, y, x, input_desc);
						out_mem[out_idx] = (Type)(cached_exp_vals[c] / Z);
					}
				}

		return output;
	}

	virtual memory generate_reference(const std::vector<memory> & inputs) override
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

	static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<test_params*, cldnn::primitive*>>& info)
	{
		std::stringstream res;

		const auto & p = std::get<0>(info.param);

		assert(p->data_type == data_types::f32 ||
			p->data_type == data_types::f16);

		res << info.index
			<< "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

		for (unsigned i = 0; i < p->input_layouts.size(); ++i)
		{
			const auto chans = format::traits(p->fmt).order;

			res << "_" << "Input" << i;
			for (unsigned int j = 0; j < p->input_layouts[i].size.sizes(p->fmt).size(); ++j)
			{
				res << chans[j] << p->input_layouts[i].size.sizes(p->fmt)[j];
			}
		}

		return res.str();
	}

private:

	static std::vector<tests::test_params*> all_generic_params;
	static std::vector<cldnn::primitive*> all_layer_params;

};