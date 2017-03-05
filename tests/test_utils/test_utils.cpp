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

#include <api/memory.hpp>
#include <api/primitive.hpp>
#include <api/primitives/input_layout.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils.h"

using namespace cldnn;

namespace tests 
{
	generic_test::generic_test() : generic_params(std::get<0>(GetParam())), layer_parmas(std::get<1>(GetParam()))
	{}

	void generic_test::TearDownTestCase()
	{
		for (auto generic_params : all_generic_params)
		{
			delete generic_params;
		}
	}

	void generic_test::run_single_test()
	{
		engine engine;

		auto input = memory::allocate(engine, { data_types::f32, (*generic_params).input });
		auto output_ref = memory::allocate(engine, input.get_layout());

		tests::set_random_values<float>(input, -100, 100);

		topology topology;
		topology.add(input_layout("input", input.get_layout()));
		topology.add(*layer_parmas);

		network network(engine, topology);

		network.set_input_data("input", input);

		auto outputs = network.execute();
		EXPECT_EQ(outputs.size(), size_t(1));
		EXPECT_EQ(outputs.begin()->first, "lrn");

		auto output = outputs.begin()->second.get_memory();

		generate_reference(input, output_ref);

		auto out_res = output.pointer<float>();
		auto out_ref = output_ref.pointer<float>();
		for (size_t i = 0; i < input.get_layout().count(); ++i)
		{
			EXPECT_NEAR(out_res[i], out_ref[i], 1e-4);
			if (HasFailure())
			{
				break;
			}
		}
	}

	std::vector<test_params*> generic_test::generate_generic_test_params()
	{
		std::vector<int32_t> batch_sizes = { 1, 2 };// 4, 8, 16};
		std::vector<int32_t> feature_sizes = { 1, 2 };// , 3, 15};
		std::vector<tensor> input_sizes = { { format::yx,{ 100,100 } } ,{ format::yx,{ 227,227 } } };
		//, { format::yx,{ 400,600 } } , { format::yx,{ 531,777 } } , { format::yx,{ 4096,1980 } } ,
		//{ format::yx,{ 1,1 } } , { format::yx,{ 2,2 } } , { format::yx,{ 3,3 } } , { format::yx,{ 4,4 } } , { format::yx,{ 5,5 } } , { format::yx,{ 6,6 } } , { format::yx,{ 7,7 } } ,
		//{ format::yx,{ 8,8 } } , { format::yx,{ 9,9 } } , { format::yx,{ 10,10 } } , { format::yx,{ 11,11 } } , { format::yx,{ 12,12 } } , { format::yx,{ 13,13 } } ,
		//{ format::yx,{ 14,14 } } , { format::yx,{ 15,15 } } , { format::yx,{ 16,16 } } };

		for (int batch_size : batch_sizes)
		{
			for (int feature_size : feature_sizes)
			{
				for (tensor input_size : input_sizes)
				{
					all_generic_params.push_back(new test_params(batch_size, feature_size, input_size));
				}
			}
		}

		return all_generic_params;
	}

	std::vector<tests::test_params*> tests::generic_test::all_generic_params = {};
}