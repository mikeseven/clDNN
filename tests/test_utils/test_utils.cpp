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

#include "api/CPP/memory.hpp"
#include <api/CPP/primitive.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils.h"
#include "float16.h"

using namespace cldnn;

namespace tests 
{
	generic_test::generic_test() : generic_params(std::get<0>(GetParam())), layer_params(std::get<1>(GetParam())), max_ulps_diff_allowed(4)
	{
	}

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
		if (layer_params->input[0] == "reorder0")
		{
			// Add reorder layer with output padding as input to the tested layer.
			topology.add(reorder("reorder0", "input0", input_mems[0].get_layout(), "", { format::yx,{ 0, 0 } }, { format::yx,{ 3, 1 },{ 2, 5 } }));
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
        
		if (output.get_layout().data_type == data_types::f32)
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

		cldnn::format base_fmt = (cldnn::format::traits(out_layout.size.format).internal_order == "bfxy") ? cldnn::format::bfyx : cldnn::format::oiyx;

		EXPECT_EQ(out_layout.size.transform(base_fmt, 1), ref_layout.size.transform(base_fmt, 1));
		EXPECT_EQ(out_layout.data_type, ref_layout.data_type);
		EXPECT_EQ(get_expected_output_tensor(), out_layout.size);
		EXPECT_EQ(out_layout.size.get_linear_size(), ref_layout.size.get_linear_size());

		auto output_sizes = out_layout.size.transform(base_fmt, 0).sizes();

		int batch_size = output_sizes[0];
		int feature_size = output_sizes[1];
		int y_size = output_sizes[2];
		int x_size = output_sizes[3];

		auto res_data = out.pointer<Type>();
		auto ref_data = ref.pointer<Type>();

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

						EXPECT_TRUE(floating_point_equal(res_data[res_index], ref_data[ref_index], max_ulps_diff_allowed))
							<< "Expected " << (float)res_data[res_index] << " to be almost equal (within " << max_ulps_diff_allowed << " ULP's) to " << (float)ref_data[ref_index]
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

	//TODO: is it ok that it assumes flat memory?
	size_t generic_test::get_linear_index(const layout & layout, int b, int f, int y, int x)
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
			case format::byxf:
			{
				//b=sizes[0], y=sizes[1], x=sizes[2], f=sizes[3]
				fPitch = 1;
				xPitch = layout.size.sizes()[3] * fPitch;
				yPitch = layout.size.sizes()[2] * xPitch;
				bPitch = layout.size.sizes()[1] * yPitch;
				return ((b * bPitch) + (f * fPitch) + (y * yPitch) + (x * xPitch));
			}
			case format::oiyx:
			{
				//o=sizes[0], i=sizes[1], y=sizes[2], x=sizes[3]
				xPitch = 1;
				yPitch = layout.size.sizes()[3] * xPitch;
				fPitch = layout.size.sizes()[2] * yPitch;
				bPitch = layout.size.sizes()[1] * fPitch;
				return ((b * bPitch) + (f * fPitch) + (y * yPitch) + (x * xPitch));
			}
			case format::yxoi:
			{
				//y=sizes[0], x=sizes[1], o=sizes[2], i=sizes[3]
				fPitch = 1;
				bPitch = layout.size.sizes()[3] * fPitch;
				xPitch = layout.size.sizes()[2] * bPitch;
				yPitch = layout.size.sizes()[1] * xPitch;
				return ((b * bPitch) + (f * fPitch) + (y * yPitch) + (x * xPitch));
			}
			case format::oyxi:
			{
				//o=sizes[0], y=sizes[1], x=sizes[2], i=sizes[3]
				fPitch = 1;
				xPitch = layout.size.sizes()[3] * fPitch;
				yPitch = layout.size.sizes()[2] * xPitch;
				bPitch = layout.size.sizes()[1] * yPitch;
				return ((b * bPitch) + (f * fPitch) + (y * yPitch) + (x * xPitch));
			}
			case format::yxio:
			{
				//y=sizes[0], x=sizes[1], i=sizes[2], o=sizes[3]
				bPitch = 1;
				fPitch = layout.size.sizes()[3] * bPitch;
				xPitch = layout.size.sizes()[2] * fPitch;
				yPitch = layout.size.sizes()[1] * xPitch;
				return ((b * bPitch) + (f * fPitch) + (y * yPitch) + (x * xPitch));
			}
			default:
			{
				throw std::runtime_error("Format not supported yet.");
			}
		}
	}

	//TODO: change the sig to take the layout size only for the output stuff
	//TODO: is it ok that it assumes flat memory?
    size_t generic_test::get_linear_index_with_broadcast(const layout & in_layout, int b, int f, int y, int x, const layout & out_layout)
	{
		assert(in_layout.size.format == out_layout.size.format);	//TODO: won't be needed after sig change. we could support different layouts but there's no need, atm.

		const auto in0 = in_layout.size.sizes()[0];
		const auto in1 = in_layout.size.sizes()[1];
		const auto in2 = in_layout.size.sizes()[2];
		const auto in3 = in_layout.size.sizes()[3];

		const auto out0 = out_layout.size.sizes()[0];
		const auto out1 = out_layout.size.sizes()[1];
		const auto out2 = out_layout.size.sizes()[2];
		const auto out3 = out_layout.size.sizes()[3];

		assert(in0 == 1 || in0 == out0);
		assert(in1 == 1 || in1 == out1);
		assert(in2 == 1 || in2 == out2);
		assert(in3 == 1 || in3 == out3);

		uint32_t bPitch, fPitch, yPitch, xPitch;
		switch (in_layout.size.format)
		{
			case format::bfyx:
			{
				//b=sizes[0], f=sizes[1], y=sizes[2], x=sizes[3]
				xPitch = 1;
				yPitch = in3 * xPitch;
				fPitch = in2 * yPitch;
				bPitch = in1 * fPitch;
				return	(in3 == out3 ? x * xPitch : 0) +
					(in2 == out2 ? y * yPitch : 0) +
					(in1 == out1 ? f * fPitch : 0) +
					(in0 == out0 ? b * bPitch : 0);
			}
			case format::yxfb:
			{
				//y=sizes[0], x=sizes[1], f=sizes[2], b=sizes[3]
				bPitch = 1;
				fPitch = in3 * bPitch;
				xPitch = in2 * fPitch;
				yPitch = in1 * xPitch;
				return	(in3 == out3 ? b * bPitch : 0) +
					(in2 == out2 ? f * fPitch : 0) +
					(in1 == out1 ? x * xPitch : 0) +
					(in0 == out0 ? y * yPitch : 0);
			}
			case format::fyxb:
			{
				//f=sizes[0], y=sizes[1], x=sizes[2], b=sizes[3]
				bPitch = 1;
				xPitch = in3 * bPitch;
				yPitch = in2 * xPitch;
				fPitch = in1 * yPitch;
				return	(in3 == out3 ? b * bPitch : 0) +
					(in2 == out2 ? x * xPitch : 0) +
					(in1 == out1 ? y * yPitch : 0) +
					(in0 == out0 ? f * fPitch : 0);
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
		return generic_params->input_layouts[0].add(layer_params->output_padding.lower_size()).add(layer_params->output_padding.upper_size());
	}

	std::vector<test_params*> generic_test::generate_generic_test_params(std::vector<test_params*>& all_generic_params, bool use_weight_formats)
	{
		// , { format::yx,{ 531,777 } } , { format::yx,{ 4096,1980 } } ,
		//{ format::yx,{ 1,1 } } , { format::yx,{ 2,2 } } , { format::yx,{ 3,3 } } , { format::yx,{ 4,4 } } , { format::yx,{ 5,5 } } , { format::yx,{ 6,6 } } , { format::yx,{ 7,7 } } ,
		//{ format::yx,{ 8,8 } } , { format::yx,{ 9,9 } } , { format::yx,{ 10,10 } } , { format::yx,{ 11,11 } } , { format::yx,{ 12,12 } } , { format::yx,{ 13,13 } } ,
		//{ format::yx,{ 14,14 } } , { format::yx,{ 15,15 } } , { format::yx,{ 16,16 } } };
		std::vector<cldnn::format> all_test_formats = { test_input_formats };
		if (use_weight_formats)
		{
			all_test_formats.insert(all_test_formats.end(), test_weight_formats.begin(), test_weight_formats.end());
		}

		for (cldnn::data_types data_type : test_data_types)
		{
			for (cldnn::format fmt : all_test_formats)
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

	bool test_params::is_weight_format(cldnn::format fmt)
	{
		assert((cldnn::format::traits(fmt).internal_order == "bfxy") || (cldnn::format::traits(fmt).internal_order == "?oixy"));

		return cldnn::format::traits(fmt).internal_order == "?oixy";
	}
    
    std::vector<cldnn::data_types> generic_test::test_data_types = { cldnn::data_types::f32 , cldnn::data_types::f16 };
    std::vector<cldnn::format> generic_test::test_input_formats = { cldnn::format::bfyx , cldnn::format::yxfb, cldnn::format::fyxb, cldnn::format::byxf };
	std::vector<cldnn::format> generic_test::test_weight_formats = { cldnn::format::oiyx , cldnn::format::yxoi , cldnn::format::oyxi, cldnn::format::yxio };
    std::vector<int32_t> generic_test::test_batch_sizes = { 1, 2 };// 4, 8, 16};
    std::vector<int32_t> generic_test::test_feature_sizes = { 1, 2 };// , 3, 15};
    std::vector<tensor> generic_test::test_input_sizes = { { format::yx,{ 100,100 } } ,{ format::yx,{ 227,227 } } ,{ format::yx,{ 400,600 } } };    
}