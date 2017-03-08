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

	void generic_test::run_single_test()
	{
		auto input = memory::allocate(engine, { data_types::f32, (*generic_params).input });

		tests::set_random_values<float>(input, -100, 100);

		topology topology;
		topology.add(input_layout("input", input.get_layout()));
		topology.add(*layer_parmas);

		if (!is_format_supported(input.get_layout().size.format))
		{
			ASSERT_THROW(network bad(engine, topology), std::runtime_error);
			return;
		}

		network network(engine, topology);

		network.set_input_data("input", input);

		auto outputs = network.execute();
		EXPECT_EQ(outputs.size(), size_t(1));

		auto output = outputs.begin()->second.get_memory();

		auto output_ref = generate_reference(input);

		auto out_layout = output.get_layout();
		auto ref_layout = output_ref.get_layout();

		assert(out_layout.size.transform(cldnn::format::bfyx, 0) == ref_layout.size.transform(cldnn::format::bfyx, 0));

		int batch_size = out_layout.size.transform(cldnn::format::bfyx, 0).sizes()[0];
		int feature_size = out_layout.size.transform(cldnn::format::bfyx, 0).sizes()[1];
		int y_size = out_layout.size.transform(cldnn::format::bfyx, 0).sizes()[2];
		int x_size = out_layout.size.transform(cldnn::format::bfyx, 0).sizes()[3];

		auto res_data = output.pointer<float>().data();
		auto ref_data = output_ref.pointer<float>().data();

		for (int b = 0; b < batch_size; b++)
		{
			for (int f = 0; f < feature_size; f++)
			{
				for (int y = 0; y < y_size; y++)
				{
					for (int x = 0; x < x_size; x++)
					{
						size_t res_index = get_linear_index(out_layout,b,f,y,x);
						size_t ref_index = get_linear_index(ref_layout,b,f,y,x);

						EXPECT_NEAR(res_data[res_index], ref_data[ref_index], 1e-4);
						if (HasFailure())
						{
							break;
						}
					}
					if (HasFailure())
					{
						break;
					}
				}
			}
			if (HasFailure())
			{
				break;
			}
		}
	}

	uint32_t generic_test::get_linear_index(layout layout, int b, int f, int y, int x)
	{
		uint32_t bPitch, fPitch, yPitch, xPitch;
		switch (layout.size.format)
		{
			case format::bfyx:
			{
				//b=sizes[0], f=sizes[1], y=sizes[2], x=sizes[3]
				xPitch = 1;
				yPitch = layout.size.sizes()[3] * xPitch;
				fPitch = layout.size.sizes()[2] * yPitch;
				bPitch = layout.size.sizes()[1] * fPitch;
				return ((b * bPitch) + (f * fPitch) + (y * yPitch) + (x * xPitch));
			}
			case format::yxfb:
			{
				//y=sizes[0], x=sizes[1], f=sizes[2], b=sizes[3]
				bPitch = 1;
				fPitch = layout.size.sizes()[3] * bPitch;
				xPitch = layout.size.sizes()[2] * fPitch;
				yPitch = layout.size.sizes()[1] * xPitch;
				return ((b * bPitch) + (f * fPitch) + (y * yPitch) + (x * xPitch));
			}
			case format::fyxb:
			{
				//f=sizes[0], y=sizes[1], x=sizes[2], b=sizes[3]
				bPitch = 1;
				xPitch = layout.size.sizes()[3] * bPitch;
				yPitch = layout.size.sizes()[2] * xPitch;
				fPitch = layout.size.sizes()[1] * yPitch;
				return ((b * bPitch) + (f * fPitch) + (y * yPitch) + (x * xPitch));
			}
			default:
			{
				throw std::runtime_error("Format not supported yet.");
			}
		}
	}


	std::vector<test_params*> generic_test::generate_generic_test_params(std::vector<test_params*> all_generic_params)
	{
		std::vector<cldnn_format_type> formats = { cldnn_format_type::cldnn_format_bfyx , cldnn_format_type::cldnn_format_yxfb, cldnn_format_type::cldnn_format_fyxb };
		std::vector<int32_t> batch_sizes = { 1, 2 };// 4, 8, 16};
		std::vector<int32_t> feature_sizes = { 1, 2 };// , 3, 15};
		std::vector<tensor> input_sizes = { { format::yx,{ 100,100 } } ,{ format::yx,{ 227,227 } } ,{ format::yx,{ 400,600 } } };
		// , { format::yx,{ 531,777 } } , { format::yx,{ 4096,1980 } } ,
		//{ format::yx,{ 1,1 } } , { format::yx,{ 2,2 } } , { format::yx,{ 3,3 } } , { format::yx,{ 4,4 } } , { format::yx,{ 5,5 } } , { format::yx,{ 6,6 } } , { format::yx,{ 7,7 } } ,
		//{ format::yx,{ 8,8 } } , { format::yx,{ 9,9 } } , { format::yx,{ 10,10 } } , { format::yx,{ 11,11 } } , { format::yx,{ 12,12 } } , { format::yx,{ 13,13 } } ,
		//{ format::yx,{ 14,14 } } , { format::yx,{ 15,15 } } , { format::yx,{ 16,16 } } };

		for (cldnn_format_type fmt : formats)
		{
			for (int batch_size : batch_sizes)
			{
				for (int feature_size : feature_sizes)
				{
					for (tensor input_size : input_sizes)
					{
						all_generic_params.push_back(new test_params(fmt, batch_size, feature_size, input_size));
					}
				}
			}
		}

		return all_generic_params;
	}
}