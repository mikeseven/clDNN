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
#include "api/primitives/normalization.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <iostream>
#include "float16.h"

TEST(local_response_normalization_gpu, lrn_test) {

    using namespace cldnn;
    using namespace tests;

    // test initialization

    // input-output parameters:

    const int32_t px = 2, py = 2, pb = 1, pf = 7, psize = 3;

    std::initializer_list<float> input_oracle_init = {
        -1.0f, -0.5f,  0.0f,  0.5f,  1.0f,  1.5f,  2.0f,    // b=0, x=0, y=0
        -2.0f, -1.7f, -1.2f, -0.7f, -0.2f,  0.3f,  0.8f,    // b=0, x=1, y=0
        0.1f,  0.4f,  0.9f,  1.4f,  1.9f,  2.4f,  2.9f,    // b=0, x=0, y=1
        -10.0f, -8.0f, -7.5f, -7.0f, -6.5f, -6.0f, -5.5f };  // b=0, x=1, y=1

    std::initializer_list<float> output_oracle_init = {
        -0.54433f, -0.27217f,  0.00000f,  0.27217f,  0.32366f,  0.30814f,  0.45266f,    // b=0, x=0, y=0
        -0.42484f, -0.31845f, -0.32025f, -0.30941f, -0.13928f,  0.19550f,  0.53034f,    // b=0, x=1, y=0
        0.08889f,  0.23964f,  0.32244f,  0.31267f,  0.28876f,  0.26604f,  0.37728f,    // b=0, x=0, y=1
        -0.21721f, -0.13945f, -0.15913f, -0.16455f, -0.17056f, -0.17725f, -0.23420f };  // b=0, x=1, y=1

                                                                                        // lrn parameters:
    const float pk = 1.0f, palpha = 3.0f, pbeta = 0.75f;

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, { format::yxfb, { py, px, pf, pb }} });
    auto output_oracle = memory::allocate(engine, input.get_layout());

    set_values(input, input_oracle_init);
    set_values(output_oracle, output_oracle_init);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(normalization("lrn", "input", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_across_channel));

    network network(engine, topology);

    network.set_input_data("input", input);

    // ------------------------------------------------------------------------------------------------
    // test run
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "lrn");

    auto output = outputs.begin()->second.get_memory();

    // analysis of results
    bool   result = true;

    try {

        auto buff = output.pointer<float>();
        auto buff_oracle = output_oracle.pointer<float>();

        for (size_t i = 0; i < px*py*pb*pf; ++i) {
            EXPECT_NEAR(buff[i], buff_oracle[i], 1e-04F);
        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }

    EXPECT_EQ(true, result);
    // ------------------------------------------------------------------------------------------------
    // test clean

}

TEST(local_response_normalization_gpu, lrn_test_batches) {

    using namespace cldnn;
    using namespace tests;

    // test initialization

    // input-output parameters:

    const int32_t px = 2, py = 1, pb = 2, pf = 7, psize = 3;

    std::initializer_list<float> input_oracle_init = {
        -1.0f, -2.0f,
        -0.5f, -1.7f,
         0.0f, -1.2f,
         0.5f, -0.7f,
         1.0f, -0.2f,
         1.5f,  0.3f,
         2.0f,  0.8f,
        
         0.1f,  -10.0f,
         0.4f,  -8.0f,
         0.9f,  -7.5f,
         1.4f,  -7.0f,
         1.9f,  -6.5f,
         2.4f,  -6.0f,
         2.9f,  -5.5f };

    std::initializer_list<float> output_oracle_init = {
        -0.54433f, -0.42484f,
        -0.27217f, -0.31845f,
         0.00000f, -0.32025f,
         0.27217f, -0.30941f,
         0.32366f, -0.13928f,
         0.30814f,  0.19550f,
         0.45266f,  0.53034f,
        
         0.08889f, -0.21721f,
         0.23964f, -0.13945f,
         0.32244f, -0.15913f,
         0.31267f, -0.16455f,
         0.28876f, -0.17056f,
         0.26604f, -0.17725f,
         0.37728f, -0.23420f };
                                                                                        // lrn parameters:
    const float pk = 1.0f, palpha = 3.0f, pbeta = 0.75f;

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, { format::yxfb, { py, px, pf, pb } } });
    auto output_oracle = memory::allocate(engine, input.get_layout());

    set_values(input, input_oracle_init);
    set_values(output_oracle, output_oracle_init);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(normalization("lrn", "input", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_across_channel));

    network network(engine, topology);

    network.set_input_data("input", input);

    // ------------------------------------------------------------------------------------------------
    // test run
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "lrn");

    auto output = outputs.begin()->second.get_memory();

    // analysis of results
    bool   result = true;

    try {

        auto buff = output.pointer<float>();
        auto buff_oracle = output_oracle.pointer<float>();

        for (size_t i = 0; i < px*py*pb*pf; ++i) {
            EXPECT_NEAR(buff[i], buff_oracle[i], 1e-04F);
        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }

    EXPECT_EQ(true, result);
    // ------------------------------------------------------------------------------------------------
    // test clean
}

TEST(local_response_normalization_gpu, test_within_channel) {

    using namespace cldnn;
    using namespace tests;

    // test initialization

    // input-output parameters:

    const int32_t px = 3, py = 3, pb = 2, pf = 7, psize = 3;

    std::initializer_list<float> input_oracle_init = {
        1.270355f, -0.276111f, 0.302943f, -0.103556f, -0.388616f, -0.669846f, -2.441765f, 0.613500f, 0.150486f, 0.326861f, 0.945122f, -2.423861f, -1.042511f, 0.772510f, -0.052556f, -0.897190f, -1.171790f, 1.003860f, -0.343486f, -0.785948f, 0.761825f, 0.436011f, 1.426594f, -0.435498f, -0.576754f, -1.267563f, -0.391251f, 1.093607f, 0.087347f, 1.592100f, -0.298722f, -0.623962f, -2.065285f, -0.082339f, 0.979977f, 0.331055f, -0.693516f, 2.132489f, -0.924463f, 0.905520f, 1.002985f, -0.524016f, 0.839577f, -0.210728f, -0.051400f, -1.592077f, 1.420933f, -0.736530f, -0.569861f, 0.863847f, -1.238764f, -1.176492f, -0.128082f, -0.521338f, 0.483847f, 0.625554f, 0.500752f, -1.098126f, -1.255482f, -1.579891f, 1.626291f, -0.969261f, -0.480432f, 2.858260f, 0.350113f, -0.901251f, -0.506315f, 1.029405f, 1.100748f, -0.476692f, -0.191909f, 0.282852f, -0.309875f, 1.023757f, 0.340389f, -1.042515f, 1.311320f, -1.351468f, -0.785442f, 0.478464f, -1.374495f, -0.225860f, -0.632197f, -0.221852f, 0.481983f, -0.550967f, 1.679756f, -1.230878f, -0.030937f, -0.181357f, -0.079843f, -0.096342f, -0.359259f, -0.630267f, 1.346543f, -0.532399f, 0.200703f, -0.567115f, 0.968568f, -0.753102f, -0.600049f, -0.726843f, -0.289637f, -1.344441f, 0.655292f, 1.499955f, -1.280091f, 0.541340f, -1.353928f, 0.252612f, 0.337532f, -2.277788f, 0.396562f, 0.019647f, -1.600525f, 0.901632f, 0.731327f, 0.218514f, -0.135189f, 0.012691f, 0.593577f, -0.484725f, 1.189892f, 0.164477f, 0.217850f, -0.129365f
    };

    std::initializer_list<float> output_oracle_init = {
        1.104031f, -0.231370f, 0.284904f, -0.063737f, -0.233475f, -0.611322f, -1.624163f, 0.399010f, 0.139065f, 0.268757f, 0.572463f, -1.547730f, -0.754049f, 0.412919f, -0.030309f, -0.686426f, -0.847080f, 0.810033f, -0.277502f, -0.605939f, 0.598232f, 0.314710f, 0.981229f, -0.309634f, -0.433626f, -0.934810f, -0.297212f, 0.961720f, 0.053086f, 1.024645f, -0.246164f, -0.362595f, -1.266915f, -0.073635f, 0.674134f, 0.228848f, -0.453550f, 1.324480f, -0.609545f, 0.572040f, 0.603035f, -0.344734f, 0.695176f, -0.171415f, -0.046360f, -1.106191f, 0.893684f, -0.533294f, -0.369697f, 0.505783f, -0.883187f, -0.980744f, -0.095681f, -0.431848f, 0.380299f, 0.423198f, 0.365109f, -0.713527f, -0.717714f, -1.083316f, 1.088807f, -0.574096f, -0.340504f, 1.657694f, 0.187924f, -0.717123f, -0.290558f, 0.545723f, 0.869626f, -0.422233f, -0.155916f, 0.237084f, -0.235859f, 0.701691f, 0.248245f, -0.756730f, 0.792027f, -0.884943f, -0.608374f, 0.305517f, -0.953085f, -0.208910f, -0.483805f, -0.172639f, 0.400975f, -0.387163f, 1.304507f, -1.055162f, -0.022323f, -0.144725f, -0.067647f, -0.079446f, -0.304201f, -0.521436f, 1.025620f, -0.416269f, 0.166215f, -0.435614f, 0.763649f, -0.613918f, -0.461385f, -0.581237f, -0.190749f, -0.835978f, 0.469027f, 1.033966f, -0.851382f, 0.406909f, -0.869536f, 0.161385f, 0.328396f, -1.269751f, 0.215731f, 0.017277f, -0.962926f, 0.530478f, 0.651228f, 0.207326f, -0.115753f, 0.011171f, 0.559947f, -0.412489f, 1.042642f, 0.155953f, 0.186214f, -0.113503f
    };
    
    // lrn parameters:
    const float pk = 1.0f, palpha = 1.0f, pbeta = 0.75f;

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, { format::bfyx, { pb, pf, py, px }} });
    auto output_oracle = memory::allocate(engine, input.get_layout());

    set_values(input, input_oracle_init);
    set_values(output_oracle, output_oracle_init);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(normalization("lrn", "input", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_within_channel));

    network network(engine, topology);

    network.set_input_data("input", input);

    // ------------------------------------------------------------------------------------------------
    // test run
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "lrn");

    auto output = outputs.begin()->second.get_memory();

    // analysis of results
    bool result = true;

    try {

        auto buff = output.pointer<float>();
        auto buff_oracle = output_oracle.pointer<float>();

        for (int i = 0; i < px*py*pb*pf; ++i) {
            EXPECT_NEAR(buff[i], buff_oracle[i], 1e-04F);
        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }

    EXPECT_EQ(true, result);
    // ------------------------------------------------------------------------------------------------
    // test clean

}


using namespace cldnn;

class lrn_test : public tests::generic_test
{

public:

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

    void print_params()
    {
        const cldnn::normalization* lrn = (cldnn::normalization*)layer_params;
        printf("Layer params: beta %f k %f size %u norm region %d alpha %f\n", lrn->beta, lrn->k, lrn->size, lrn->norm_region, lrn->alpha);
    }

	static std::vector<cldnn::primitive*> generate_specific_test_params()
	{
		std::vector<cldnn_lrn_norm_region> norm_regions = { cldnn_lrn_norm_region_across_channel, cldnn_lrn_norm_region_within_channel };

		for (auto norm_region : norm_regions)
		{
			all_layer_params.push_back(new normalization("lrn", "input0", 3, 1.f, 5e-05f, 0.75f, norm_region));
			all_layer_params.push_back(new normalization("lrn", "input0", 3, 1.f, 5e-05f, 0.75f, norm_region, { format::yx,{ 0, 0 } }, { format::yx,{ 13, 6 } }));
			all_layer_params.push_back(new normalization("lrn", "input0", 5, 17.19f, 0.079f, 0.19f, norm_region));
			all_layer_params.push_back(new normalization("lrn", "input0", 5, 17.19f, 0.079f, 0.19f, norm_region, { format::yx,{ 0, 0 } }, { format::yx,{ 5, 11 },{ 19, 0 } }));
		}

		//The test checks only valid combinations.
		//TODO: add more combinations.

		return all_layer_params;
	}

	static std::vector<tests::test_params*> generate_generic_test_params()
	{
		return generic_test::generate_generic_test_params(all_generic_params);
	}

	virtual bool is_format_supported(cldnn::format format)
	{
		return ((format == cldnn_format_type::cldnn_format_yxfb) || (format == cldnn_format_type::cldnn_format_bfyx));
	}

	template<typename Type>
	memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
	{
		const cldnn::normalization* lrn = (cldnn::normalization*)layer_params;

		//Output is bfyx
        data_types dt = inputs[0].get_layout().data_type;
		auto output = memory::allocate( engine, cldnn::layout(dt, inputs[0].get_layout().size.add(lrn->output_padding().lower_size()).add(lrn->output_padding().upper_size()).transform(cldnn::format::bfyx, 0)) );

		// TODO: need to add support for input padding.
		assert(!lrn->input_padding());

		Type beta = lrn->beta;
		Type k = lrn->k;
		uint32_t size = lrn->size;
		cldnn_lrn_norm_region lrn_norm_region = lrn->norm_region;
		Type alpha_sign = std::signbit(lrn->alpha) ? -1.0f : 1.0f;
		Type alpha = (dt == cldnn::data_types::f32) ? (Type)lrn->alpha : alpha_sign;
		Type alpha_div_by_size = (dt == cldnn::data_types::f32) ? (Type)(lrn->alpha / lrn->size) : alpha_sign;
		Type alpha_div_by_size_abs_sqrt = (dt == cldnn::data_types::f32) ? 1.0f : std::sqrt(std::abs(lrn->alpha / lrn->size));
		Type alpha_abs_sqrt = (dt == cldnn::data_types::f32) ? 1.0f : std::sqrt(std::abs(lrn->alpha));

		Type* input_mem = inputs[0].pointer<Type>().data();
		Type* output_mem = output.pointer<Type>().data();

		int batch = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0];
		int feature = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1];
		int height = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2];
		int width = inputs[0].get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3];

		int output_height = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2];
		int output_width = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3];

		//Initialized output with zeros.
		memset(output_mem, 0, output.get_layout().data_size());

		switch (lrn_norm_region)
		{
			case cldnn_lrn_norm_region::cldnn_lrn_norm_region_across_channel:
			{
				for (int n = 0; n < batch; ++n) 
				{
					for (int c = 0; c < feature; ++c) 
					{
						for (int h = 0; h < height; ++h) 
						{
							for (int w = 0; w < width; ++w) 
							{
								int c_start = c - (size - 1) / 2;
								int c_end = std::min((int)(c_start + size), feature);
								c_start = std::max(c_start, 0);
								Type scale = 0;
								for (int i = c_start; i < c_end; ++i) 
								{
									int input_index = get_linear_index(inputs[0].get_layout(), n, i, h, w);
									Type value = input_mem[input_index] * alpha_div_by_size_abs_sqrt;
									scale += value * value;
								}
								scale = scale * alpha_div_by_size + k;

								int output_index = (n * feature + c) * output_height * output_width;
								tensor lower_padding = lrn->output_padding().lower_size().transform(cldnn::format::bfyx, 0);
								output_index += (lower_padding.sizes()[2] + h) * output_width + lower_padding.sizes()[3] + w;

								int input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w);
								output_mem[output_index] = input_mem[input_index] * (Type)(float)pow((float)scale, -(float)beta);
							}
						}
					}
				}
				break;
			}
			case cldnn_lrn_norm_region::cldnn_lrn_norm_region_within_channel:
			{
				int pad = (size - 1) / 2;
				for (int n = 0; n < batch; ++n) 
				{
					for (int c = 0; c < feature; ++c) 
					{
						for (int h = 0; h < height; ++h) 
						{
							for (int w = 0; w < width; ++w) 
							{
								Type scale = 0.f;
								int h_start = h - pad;
								int w_start = w - pad;
								int h_end = std::min((int)(h_start + size), height + pad);
								int w_end = std::min((int)(w_start + size), width + pad);
								int pool_size = (h_end - h_start) * (w_end - w_start);
								h_start = std::max(h_start, 0);
								w_start = std::max(w_start, 0);
								h_end = std::min(h_end, height);
								w_end = std::min(w_end, width);
								for (int nh = h_start; nh < h_end; ++nh) 
								{
									for (int nw = w_start; nw < w_end; ++nw) 
									{
										int input_index = get_linear_index(inputs[0].get_layout(), n, c, nh, nw);
										Type value = input_mem[input_index] * alpha_abs_sqrt;
										scale += value * value;
									}
								}
								scale /= pool_size;
								int input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w);

								int output_index = (n * feature + c) * output_height * output_width;
								tensor lower_padding = lrn->output_padding().lower_size().transform(cldnn::format::bfyx, 0);
								output_index += (lower_padding.sizes()[2] + h) * output_width + lower_padding.sizes()[3] + w;

								output_mem[output_index] = input_mem[input_index] * (Type)(float)pow((float)(scale * alpha + k), -(float)beta);							
							}
						}
					}
				}
				break;
			}
			default:
			{
				assert(0);
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

std::vector<cldnn::primitive*> lrn_test::all_layer_params = {};
std::vector<tests::test_params*> lrn_test::all_generic_params = {};

TEST_P(lrn_test, DISABLED_test_all)
{
	run_single_test();
}

INSTANTIATE_TEST_CASE_P(LRN, 
						lrn_test, 
						::testing::Combine(::testing::ValuesIn(lrn_test::generate_generic_test_params()),
										   ::testing::ValuesIn(lrn_test::generate_specific_test_params())), 
						tests::generic_test::custom_param_name_functor());

