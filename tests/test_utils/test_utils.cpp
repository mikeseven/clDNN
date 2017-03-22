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
#include "float16.h"

using namespace cldnn;

namespace tests 
{
	generic_test::generic_test() : generic_params(std::get<0>(GetParam())), layer_params(std::get<1>(GetParam()))
	{}

	void generic_test::run_single_test()
	{
        assert((generic_params->data_type == data_types::f32) || (generic_params->data_type == data_types::f16));
        
        topology topology;               
        topology.add(*layer_params);
        
        std::vector<memory> input_mems;

        for (size_t i = 0 ; i < generic_params->input_layouts.size() ; i++)
        {           
            input_mems.push_back( memory::allocate(engine, { generic_params->data_type, generic_params->input_layouts[i] }) );
            
            if (generic_params->data_type == data_types::f32)
            {
                tests::set_random_values<float>(input_mems[i], -100, 100);
            }
            else
            {
                tests::set_random_values<FLOAT16>(input_mems[i], -100, 100);
            }   
            
            std::string layer_name = "input" + std::to_string(i);
            topology.add(input_layout(layer_name, input_mems[i].get_layout()));            
            
            if (!is_format_supported(generic_params->input_layouts[i].format))
            {
                ASSERT_THROW(network bad(engine, topology), std::exception);
                return;
            }       
        }
        		
		network network(engine, topology);

        for (size_t i = 0 ; i < generic_params->input_layouts.size() ; i++)
        {
            std::string layer_name = "input" + std::to_string(i);
            network.set_input_data(layer_name, input_mems[i]);
        }		
        
        prepare_input_for_test(input_mems);

		auto outputs = network.execute();
		EXPECT_EQ(outputs.size(), size_t(1));

		auto output = outputs.begin()->second.get_memory();

		auto output_ref = generate_reference(input_mems);
        
		if (generic_params->data_type == data_types::f32)
		{
			compare_buffers<float>(output, output_ref);
		}
		else
		{
			compare_buffers<FLOAT16>(output, output_ref);
		}   
	}

	template<typename Type>
	void generic_test::compare_buffers(const memory& out, const memory& ref)
	{
		auto out_layout = out.get_layout();
		auto ref_layout = ref.get_layout();

		EXPECT_EQ(out_layout.size.transform(cldnn::format::bfyx, 0), ref_layout.size.transform(cldnn::format::bfyx, 0));
		EXPECT_EQ(out_layout.data_type, ref_layout.data_type);
		EXPECT_EQ(get_expected_output_tensor(), out_layout.size);
		EXPECT_EQ(out_layout.size.get_linear_size(), ref_layout.size.get_linear_size());

		int batch_size = out_layout.size.transform(cldnn::format::bfyx, 0).sizes()[0];
		int feature_size = out_layout.size.transform(cldnn::format::bfyx, 0).sizes()[1];
		int y_size = out_layout.size.transform(cldnn::format::bfyx, 0).sizes()[2];
		int x_size = out_layout.size.transform(cldnn::format::bfyx, 0).sizes()[3];

		auto res_data = out.pointer<Type>().data();
		auto ref_data = ref.pointer<Type>().data();

		for (int b = 0; b < batch_size; b++)
		{
			for (int f = 0; f < feature_size; f++)
			{
				for (int y = 0; y < y_size; y++)
				{
					for (int x = 0; x < x_size; x++)
					{
						size_t res_index = get_linear_index(out_layout, b, f, y, x);
						size_t ref_index = get_linear_index(ref_layout, b, f, y, x);

						EXPECT_TRUE(floating_point_equal(res_data[res_index], ref_data[ref_index]))
							<< "Expected " << (float)res_data[res_index] << " to be almost equal (within 4 ULP's) to " << (float)ref_data[ref_index] 
							<< " (ref index = " << ref_index << ", B " << b << ", F "<< f << ", Y " << y << ", X " << x << ")!";

						if (HasFailure())
						{
							return;
						}
					}
				}
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

	//Default implementation. Should be overridden in derived class otherwise.
	cldnn::tensor generic_test::get_expected_output_tensor()
	{
		return generic_params->input_layouts[0].add(layer_params->output_padding().lower_size()).add(layer_params->output_padding().upper_size());
	}

	std::vector<test_params*> generic_test::generate_generic_test_params(std::vector<test_params*> all_generic_params)
	{
		// , { format::yx,{ 531,777 } } , { format::yx,{ 4096,1980 } } ,
		//{ format::yx,{ 1,1 } } , { format::yx,{ 2,2 } } , { format::yx,{ 3,3 } } , { format::yx,{ 4,4 } } , { format::yx,{ 5,5 } } , { format::yx,{ 6,6 } } , { format::yx,{ 7,7 } } ,
		//{ format::yx,{ 8,8 } } , { format::yx,{ 9,9 } } , { format::yx,{ 10,10 } } , { format::yx,{ 11,11 } } , { format::yx,{ 12,12 } } , { format::yx,{ 13,13 } } ,
		//{ format::yx,{ 14,14 } } , { format::yx,{ 15,15 } } , { format::yx,{ 16,16 } } };

		for (cldnn::data_types data_type : test_data_types)
		{
			for (cldnn::format fmt : test_formats)
			{
				for (int batch_size : test_batch_sizes)
				{
					for (int feature_size : test_feature_sizes)
					{
						for (tensor input_size : test_input_sizes)
						{
							all_generic_params.push_back(new test_params(data_type, fmt, batch_size, feature_size, input_size));
						}
					}
				}
			}
		}		

		return all_generic_params;
	}

	std::string test_params::print_tensor(cldnn::tensor t)
	{
		std::stringstream str;
		str << "Format " << format::traits(t.format).order << " sizes [ ";
		for (size_t i = 0; i < t.sizes().size(); i++)
		{
			str << t.sizes()[i] << " ";
		}
		str << "]";
		return str.str();
	}
    
	std::string test_params::print()
    {
		std::stringstream str;
		str << "Data type: " << data_type_traits::name(data_type) << std::endl;

        for (int j = 0 ; j < (int)input_layouts.size(); j++)
        {
            const cldnn::tensor& t = input_layouts[j];
            
			str << "Input " << j << ": " << print_tensor(t) << std::endl;
        }
		return str.str();
    }
    
    std::vector<cldnn::data_types> generic_test::test_data_types = { cldnn::data_types::f32, cldnn::data_types::f16 };
    std::vector<cldnn::format> generic_test::test_formats = { cldnn::format::bfyx , cldnn::format::yxfb, cldnn::format::fyxb };
    std::vector<int32_t> generic_test::test_batch_sizes = { 1, 2 };// 4, 8, 16};
    std::vector<int32_t> generic_test::test_feature_sizes = { 1, 2 };// , 3, 15};
    std::vector<tensor> generic_test::test_input_sizes = { { format::yx,{ 100,100 } } ,{ format::yx,{ 227,227 } } ,{ format::yx,{ 400,600 } } };    
}