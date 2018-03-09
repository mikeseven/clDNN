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
#include "api/CPP/lrn.hpp"
#include "api/CPP/reorder.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <iostream>
#include "float16.h"


TEST(local_response_normalization_gpu_yxfb_output_padding, lrn_test) {

    using namespace cldnn;
    using namespace tests;

    // test initialization

    // input-output parameters:

    const int32_t x = 2, y = 2, b = 1, f = 7, size = 3;

    const int32_t out_padding_x = 1;
    const int32_t out_padding_y = 1;

    std::initializer_list<float> input_oracle_init = {
        -1.0f, -0.5f,  0.0f,  0.5f,  1.0f,  1.5f,  2.0f,    // b=0, x=0, y=0
        -2.0f, -1.7f, -1.2f, -0.7f, -0.2f,  0.3f,  0.8f,    // b=0, x=1, y=0
        0.1f,  0.4f,  0.9f,  1.4f,  1.9f,  2.4f,  2.9f,    // b=0, x=0, y=1
        -10.0f, -8.0f, -7.5f, -7.0f, -6.5f, -6.0f, -5.5f };  // b=0, x=1, y=1

    std::initializer_list<float> output_oracle_init = {
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,

        0,0,0,0,0,0,0,
        -0.54433f, -0.27217f,  0.00000f,  0.27217f,  0.32366f,  0.30814f,  0.45266f,    // b=0, x=0, y=0
        -0.42484f, -0.31845f, -0.32025f, -0.30941f, -0.13928f,  0.19550f,  0.53034f,    // b=0, x=1, y=0
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0.08889f,  0.23964f,  0.32244f,  0.31267f,  0.28876f,  0.26604f,  0.37728f,    // b=0, x=0, y=1
        -0.21721f, -0.13945f, -0.15913f, -0.16455f, -0.17056f, -0.17725f, -0.23420f, // b=0, x=1, y=1
        0,0,0,0,0,0,0,
        
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
    };

    // lrn parameters:
    const float k = 1.0f, alpha = 3.0f, beta = 0.75f;

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ b, f, x, y } });
    cldnn::layout out_layout = cldnn::layout(cldnn::data_types::f32, format::yxfb, {b, f, x + 2 * out_padding_x, y + 2 * out_padding_y});
    auto output_oracle = memory::allocate(engine, out_layout);

    set_values(input, input_oracle_init);
    set_values(output_oracle, output_oracle_init);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(lrn("lrn", "input", size, k, alpha, beta, cldnn_lrn_norm_region_across_channel, { { 0, 0, out_padding_x, out_padding_y } ,0 }));

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

        for (size_t i = 0; i < (x + 2 * out_padding_x)*(y + 2 * out_padding_y)*b*f; ++i) {
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

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { pb, pf, px, py } });
    auto output_oracle = memory::allocate(engine, input.get_layout());

    set_values(input, input_oracle_init);
    set_values(output_oracle, output_oracle_init);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(lrn("lrn", "input", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_across_channel));

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

TEST(local_response_normalization_gpu, lrn_input_padding_yxfb_across_channel_test) {

    using namespace cldnn;
    using namespace tests;

    // input-output parameters:
    const int32_t px = 2, py = 2, pb = 1, pf = 7, psize = 3;

    // lrn parameters:
    const float pk = 1.0f, palpha = 3.0f, pbeta = 0.75f;
 
    engine engine;
    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ pb, pf, px, py } });
    set_values(input, {
        -1.0f, -0.5f,  0.0f,  0.5f,  1.0f,  1.5f,  2.0f,    // b=0, x=0, y=0
        -2.0f, -1.7f, -1.2f, -0.7f, -0.2f,  0.3f,  0.8f,    // b=0, x=1, y=0
        0.1f,  0.4f,  0.9f,  1.4f,  1.9f,  2.4f,  2.9f,    // b=0, x=0, y=1
        -10.0f, -8.0f, -7.5f, -7.0f, -6.5f, -6.0f, -5.5f });  // b=0, x=1, y=1);

    VF<float> output_vec = {
        -0.54433f, -0.27217f,  0.00000f,  0.27217f,  0.32366f,  0.30814f,  0.45266f,    // b=0, x=0, y=0
        -0.42484f, -0.31845f, -0.32025f, -0.30941f, -0.13928f,  0.19550f,  0.53034f,    // b=0, x=1, y=0
        0.08889f,  0.23964f,  0.32244f,  0.31267f,  0.28876f,  0.26604f,  0.37728f,    // b=0, x=0, y=1
        -0.21721f, -0.13945f, -0.15913f, -0.16455f, -0.17056f, -0.17725f, -0.23420f };  // b=0, x=1, y=1
    
    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding({ { 0,0,2,2 },0 })),
        lrn("lrn", "reorder", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_across_channel, { { 0, 0, 0, 0 }, 0 })
        );
    
    network network(engine, topology);
    network.set_input_data("input", input);


    // ------------------------------------------------------------------------------------------------
    // test run
    auto outputs = network.execute();
    auto output_memory = outputs.at("lrn").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    try {
        int y_size = output_layout.size.sizes()[3];
        int x_size = output_layout.size.sizes()[2];
        int f_size = output_layout.size.sizes()[1];
        int b_size = output_layout.size.sizes()[0];
        EXPECT_EQ(y_size, py);
        EXPECT_EQ(x_size, px);
        EXPECT_EQ(f_size, pf);
        EXPECT_EQ(b_size, pb);

        for (size_t i = 0; i < output_vec.size(); ++i) { 
            EXPECT_NEAR(output_vec[i], output_ptr[i], 1e-04F);
        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }

}

TEST(local_response_normalization_gpu, lrn_input_padding_bfyx_within_channel_test) {

    using namespace cldnn;
    using namespace tests;

    // input-output parameters:
    const int32_t px = 3, py = 3, pb = 2, pf = 7, psize = 3;

    // lrn parameters:
    const float pk = 1.0f, palpha = 1.0f, pbeta = 0.75f;
    engine engine;
    auto input = memory::allocate(engine, { data_types::f32,format::bfyx,{ pb, pf, px, py } });
    set_values(input, {
        1.270355f, -0.276111f, 0.302943f, -0.103556f, -0.388616f, -0.669846f, -2.441765f, 0.613500f, 0.150486f, 0.326861f, 0.945122f, -2.423861f, -1.042511f, 0.772510f, -0.052556f, -0.897190f, -1.171790f, 1.003860f, -0.343486f, -0.785948f, 0.761825f, 0.436011f, 1.426594f, -0.435498f, -0.576754f, -1.267563f, -0.391251f, 1.093607f, 0.087347f, 1.592100f, -0.298722f, -0.623962f, -2.065285f, -0.082339f, 0.979977f, 0.331055f, -0.693516f, 2.132489f, -0.924463f, 0.905520f, 1.002985f, -0.524016f, 0.839577f, -0.210728f, -0.051400f, -1.592077f, 1.420933f, -0.736530f, -0.569861f, 0.863847f, -1.238764f, -1.176492f, -0.128082f, -0.521338f, 0.483847f, 0.625554f, 0.500752f, -1.098126f, -1.255482f, -1.579891f, 1.626291f, -0.969261f, -0.480432f, 2.858260f, 0.350113f, -0.901251f, -0.506315f, 1.029405f, 1.100748f, -0.476692f, -0.191909f, 0.282852f, -0.309875f, 1.023757f, 0.340389f, -1.042515f, 1.311320f, -1.351468f, -0.785442f, 0.478464f, -1.374495f, -0.225860f, -0.632197f, -0.221852f, 0.481983f, -0.550967f, 1.679756f, -1.230878f, -0.030937f, -0.181357f, -0.079843f, -0.096342f, -0.359259f, -0.630267f, 1.346543f, -0.532399f, 0.200703f, -0.567115f, 0.968568f, -0.753102f, -0.600049f, -0.726843f, -0.289637f, -1.344441f, 0.655292f, 1.499955f, -1.280091f, 0.541340f, -1.353928f, 0.252612f, 0.337532f, -2.277788f, 0.396562f, 0.019647f, -1.600525f, 0.901632f, 0.731327f, 0.218514f, -0.135189f, 0.012691f, 0.593577f, -0.484725f, 1.189892f, 0.164477f, 0.217850f, -0.129365f
    });  
    VF<float> output_vec = {
        1.104031f, -0.231370f, 0.284904f, -0.063737f, -0.233475f, -0.611322f, -1.624163f, 0.399010f, 0.139065f, 0.268757f, 0.572463f, -1.547730f, -0.754049f, 0.412919f, -0.030309f, -0.686426f, -0.847080f, 0.810033f, -0.277502f, -0.605939f, 0.598232f, 0.314710f, 0.981229f, -0.309634f, -0.433626f, -0.934810f, -0.297212f, 0.961720f, 0.053086f, 1.024645f, -0.246164f, -0.362595f, -1.266915f, -0.073635f, 0.674134f, 0.228848f, -0.453550f, 1.324480f, -0.609545f, 0.572040f, 0.603035f, -0.344734f, 0.695176f, -0.171415f, -0.046360f, -1.106191f, 0.893684f, -0.533294f, -0.369697f, 0.505783f, -0.883187f, -0.980744f, -0.095681f, -0.431848f, 0.380299f, 0.423198f, 0.365109f, -0.713527f, -0.717714f, -1.083316f, 1.088807f, -0.574096f, -0.340504f, 1.657694f, 0.187924f, -0.717123f, -0.290558f, 0.545723f, 0.869626f, -0.422233f, -0.155916f, 0.237084f, -0.235859f, 0.701691f, 0.248245f, -0.756730f, 0.792027f, -0.884943f, -0.608374f, 0.305517f, -0.953085f, -0.208910f, -0.483805f, -0.172639f, 0.400975f, -0.387163f, 1.304507f, -1.055162f, -0.022323f, -0.144725f, -0.067647f, -0.079446f, -0.304201f, -0.521436f, 1.025620f, -0.416269f, 0.166215f, -0.435614f, 0.763649f, -0.613918f, -0.461385f, -0.581237f, -0.190749f, -0.835978f, 0.469027f, 1.033966f, -0.851382f, 0.406909f, -0.869536f, 0.161385f, 0.328396f, -1.269751f, 0.215731f, 0.017277f, -0.962926f, 0.530478f, 0.651228f, 0.207326f, -0.115753f, 0.011171f, 0.559947f, -0.412489f, 1.042642f, 0.155953f, 0.186214f, -0.113503f
    };

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 3, 3 }, 0 })),
        lrn("lrn", "reorder", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_within_channel, { { 0, 0, 0, 0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);


    // ------------------------------------------------------------------------------------------------
    // test run
    auto outputs = network.execute();
    auto output_memory = outputs.at("lrn").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    try {
        int y_size = output_layout.size.sizes()[3];
        int x_size = output_layout.size.sizes()[2];
        int f_size = output_layout.size.sizes()[1];
        int b_size = output_layout.size.sizes()[0];
        EXPECT_EQ(y_size, py);
        EXPECT_EQ(x_size, px);
        EXPECT_EQ(f_size, pf);
        EXPECT_EQ(b_size, pb);

        for (size_t i = 0; i < output_vec.size(); ++i) {
            EXPECT_NEAR(output_vec[i], output_ptr[i], 1e-04F);
        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }

}

TEST(local_response_normalization_gpu, lrn_input_padding_yxfb_within_channel_test) {

    using namespace cldnn;
    using namespace tests;

    // input-output parameters:
    const int32_t px = 3, py = 3, pb = 2, pf = 7, psize = 3;

    // lrn parameters:
    const float pk = 1.0f, palpha = 1.0f, pbeta = 0.75f;
    engine engine;
    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ pb, pf, px, py } });
    set_values(input, {
        1.270355f, -0.276111f, 0.302943f, -0.103556f, -0.388616f, -0.669846f, -2.441765f, 0.613500f, 0.150486f, 0.326861f, 0.945122f, -2.423861f, -1.042511f, 0.772510f, -0.052556f, -0.897190f, -1.171790f, 1.003860f, -0.343486f, -0.785948f, 0.761825f, 0.436011f, 1.426594f, -0.435498f, -0.576754f, -1.267563f, -0.391251f, 1.093607f, 0.087347f, 1.592100f, -0.298722f, -0.623962f, -2.065285f, -0.082339f, 0.979977f, 0.331055f, -0.693516f, 2.132489f, -0.924463f, 0.905520f, 1.002985f, -0.524016f, 0.839577f, -0.210728f, -0.051400f, -1.592077f, 1.420933f, -0.736530f, -0.569861f, 0.863847f, -1.238764f, -1.176492f, -0.128082f, -0.521338f, 0.483847f, 0.625554f, 0.500752f, -1.098126f, -1.255482f, -1.579891f, 1.626291f, -0.969261f, -0.480432f, 2.858260f, 0.350113f, -0.901251f, -0.506315f, 1.029405f, 1.100748f, -0.476692f, -0.191909f, 0.282852f, -0.309875f, 1.023757f, 0.340389f, -1.042515f, 1.311320f, -1.351468f, -0.785442f, 0.478464f, -1.374495f, -0.225860f, -0.632197f, -0.221852f, 0.481983f, -0.550967f, 1.679756f, -1.230878f, -0.030937f, -0.181357f, -0.079843f, -0.096342f, -0.359259f, -0.630267f, 1.346543f, -0.532399f, 0.200703f, -0.567115f, 0.968568f, -0.753102f, -0.600049f, -0.726843f, -0.289637f, -1.344441f, 0.655292f, 1.499955f, -1.280091f, 0.541340f, -1.353928f, 0.252612f, 0.337532f, -2.277788f, 0.396562f, 0.019647f, -1.600525f, 0.901632f, 0.731327f, 0.218514f, -0.135189f, 0.012691f, 0.593577f, -0.484725f, 1.189892f, 0.164477f, 0.217850f, -0.129365f
    });

    VF<float> output_vec = {
        1.05207f, -0.235424f, 0.243485f, -0.0704241f, -0.280009f, -0.555606f, -1.57872f, 0.357625f, 0.116129f, 0.272097f, 0.842014f, -1.45255f, -0.857082f, 0.646524f, -0.0434003f, -0.653022f, -0.931083f, 0.637538f, -0.201605f, -0.609137f, 0.438833f, 0.235864f, 1.0346f, -0.279345f, -0.431712f, -0.7329f, -0.295393f, 0.896247f, 0.0852391f, 1.1665f, -0.238743f, -0.449381f, -1.33245f, -0.0678454f, 0.766385f, 0.186846f, -0.550157f, 1.47011f, -0.729917f, 0.706196f, 0.820132f, -0.458733f, 0.646842f, -0.169808f, -0.0346512f, -0.984102f, 1.01917f, -0.546786f, -0.361134f, 0.461837f, -0.867031f, -0.937455f, -0.091247f, -0.307932f, 0.393894f, 0.389603f, 0.38128f, -0.763277f, -0.747376f, -0.890056f, 0.931003f, -0.676738f, -0.271827f, 1.42917f, 0.228124f, -0.554606f, -0.297291f, 0.586408f, 0.821586f, -0.292486f, -0.17227f, 0.200993f, -0.210659f, 0.688231f, 0.2135f, -0.768896f, 0.998781f, -0.702207f, -0.556561f, 0.321275f, -0.91067f, -0.175192f, -0.511689f, -0.144422f, 0.411109f, -0.47011f, 1.22166f, -0.796712f, -0.0225146f, -0.143283f, -0.0738246f, -0.052588f, -0.28208f, -0.511577f, 1.02678f, -0.468306f, 0.177524f, -0.384155f, 0.815497f, -0.639118f, -0.382273f, -0.433355f, -0.2036f, -0.994292f, 0.537237f, 0.767218f, -0.950032f, 0.42696f, -0.870274f, 0.220991f, 0.289442f, -1.53791f, 0.356248f, 0.0170727f, -1.17472f, 0.651318f, 0.575796f, 0.166988f, -0.11324f, 0.00666373f, 0.4866f, -0.429681f, 0.836644f, 0.149723f, 0.190369f, -0.09055f
    };
    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 2, 2 }, 0 })),
        lrn("lrn", "reorder", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_within_channel, { { 0, 0, 0, 0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);


    // ------------------------------------------------------------------------------------------------
    // test run
    auto outputs = network.execute();
    auto output_memory = outputs.at("lrn").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    try {
        int y_size = output_layout.size.sizes()[3];
        int x_size = output_layout.size.sizes()[2];
        int f_size = output_layout.size.sizes()[1];
        int b_size = output_layout.size.sizes()[0];
        EXPECT_EQ(y_size, py);
        EXPECT_EQ(x_size, px);
        EXPECT_EQ(f_size, pf);
        EXPECT_EQ(b_size, pb);

        for (size_t i = 0; i < output_vec.size(); ++i) {
            EXPECT_NEAR(output_vec[i], output_ptr[i], 1e-04F);
        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }

}

TEST(local_response_normalization_gpu, lrn_input_padding_bfyx_across_channel_test) {

    using namespace cldnn;
    using namespace tests;

    // input-output parameters:
    const int32_t px = 3, py = 3, pb = 2, pf = 7, psize = 3;

    // lrn parameters:
    const float pk = 1.0f, palpha = 1.0f, pbeta = 0.75f;
    engine engine;
    auto input = memory::allocate(engine, { data_types::f32,format::bfyx,{ pb, pf, px, py } });
    set_values(input, {
        1.270355f, -0.276111f, 0.302943f, -0.103556f, -0.388616f, -0.669846f, -2.441765f, 0.613500f, 0.150486f, 0.326861f, 0.945122f, -2.423861f, -1.042511f, 0.772510f, -0.052556f, -0.897190f, -1.171790f, 1.003860f, -0.343486f, -0.785948f, 0.761825f, 0.436011f, 1.426594f, -0.435498f, -0.576754f, -1.267563f, -0.391251f, 1.093607f, 0.087347f, 1.592100f, -0.298722f, -0.623962f, -2.065285f, -0.082339f, 0.979977f, 0.331055f, -0.693516f, 2.132489f, -0.924463f, 0.905520f, 1.002985f, -0.524016f, 0.839577f, -0.210728f, -0.051400f, -1.592077f, 1.420933f, -0.736530f, -0.569861f, 0.863847f, -1.238764f, -1.176492f, -0.128082f, -0.521338f, 0.483847f, 0.625554f, 0.500752f, -1.098126f, -1.255482f, -1.579891f, 1.626291f, -0.969261f, -0.480432f, 2.858260f, 0.350113f, -0.901251f, -0.506315f, 1.029405f, 1.100748f, -0.476692f, -0.191909f, 0.282852f, -0.309875f, 1.023757f, 0.340389f, -1.042515f, 1.311320f, -1.351468f, -0.785442f, 0.478464f, -1.374495f, -0.225860f, -0.632197f, -0.221852f, 0.481983f, -0.550967f, 1.679756f, -1.230878f, -0.030937f, -0.181357f, -0.079843f, -0.096342f, -0.359259f, -0.630267f, 1.346543f, -0.532399f, 0.200703f, -0.567115f, 0.968568f, -0.753102f, -0.600049f, -0.726843f, -0.289637f, -1.344441f, 0.655292f, 1.499955f, -1.280091f, 0.541340f, -1.353928f, 0.252612f, 0.337532f, -2.277788f, 0.396562f, 0.019647f, -1.600525f, 0.901632f, 0.731327f, 0.218514f, -0.135189f, 0.012691f, 0.593577f, -0.484725f, 1.189892f, 0.164477f, 0.217850f, -0.129365f
    });
    VF<float> output_vec = {
        0.904202f, -0.223807f, 0.133266f, -0.0819639f, -0.328874f, -0.602996f, -1.00744f, 0.434682f, 0.120594f, 0.228382f, 0.687334f, -1.01728f, -0.797547f, 0.472207f, -0.0454504f, -0.360986f, -0.667257f, 0.782281f, -0.256817f, -0.578072f, 0.26951f, 0.329051f, 0.846009f, -0.219972f, -0.452624f, -0.675746f, -0.300223f, 0.769336f, 0.0411916f, 0.846051f, -0.236361f, -0.352222f, -1.01555f, -0.0658132f, 0.612702f, 0.310668f, -0.359229f, 0.893102f, -0.493278f, 0.699327f, 0.6696f, -0.228464f, 0.564277f, -0.169161f, -0.04696f, -0.918135f, 0.577816f, -0.530385f, -0.36924f, 0.49352f, -0.635486f, -0.578273f, -0.103219f, -0.463896f, 0.296302f, 0.401963f, 0.419959f, -0.806074f, -0.816708f, -0.834112f, 0.858759f, -0.78768f, -0.427739f, 1.05956f, 0.273461f, -0.736289f, -0.383623f, 0.629542f, 0.651408f, -0.395802f, -0.180073f, 0.193724f, -0.114482f, 0.746581f, 0.2755f, -0.759616f, 0.771742f, -0.599905f, -0.508311f, 0.448856f, -0.936737f, -0.217568f, -0.469796f, -0.206822f, 0.343294f, -0.297064f, 0.81016f, -0.81808f, -0.027259f, -0.109758f, -0.0692847f, -0.0811869f, -0.306562f, -0.537074f, 0.719086f, -0.296942f, 0.108583f, -0.388885f, 0.744052f, -0.48417f, -0.542323f, -0.612229f, -0.130678f, -0.729556f, 0.55842f, 0.728939f, -0.783465f, 0.382556f, -0.865479f, 0.227862f, 0.291821f, -1.0317f, 0.262522f, 0.0137127f, -0.778798f, 0.572342f, 0.607173f, 0.151687f, -0.132479f, 0.0123407f, 0.270832f, -0.442052f, 0.890348f, 0.103148f, 0.180312f, -0.113984f
    };

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 2, 2 }, 0 })),
        lrn("lrn", "reorder", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_across_channel, { { 0, 0, 0, 0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);


    // ------------------------------------------------------------------------------------------------
    // test run
    auto outputs = network.execute();
    auto output_memory = outputs.at("lrn").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    try {
        int y_size = output_layout.size.sizes()[3];
        int x_size = output_layout.size.sizes()[2];
        int f_size = output_layout.size.sizes()[1];
        int b_size = output_layout.size.sizes()[0];
        EXPECT_EQ(y_size, py);
        EXPECT_EQ(x_size, px);
        EXPECT_EQ(f_size, pf);
        EXPECT_EQ(b_size, pb);

        for (size_t i = 0; i < output_vec.size(); ++i) {
            EXPECT_NEAR(output_vec[i], output_ptr[i], 1e-04F);
        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }
}

TEST(local_response_normalization_gpu, lrn_input_padding_yxfb_b8_test) {

    using namespace cldnn;
    using namespace tests;

    // input-output parameters:
    const int32_t px = 2, py = 1, pb = 8, pf = 8, psize = 3;

    // lrn parameters:
    const float pk = 1.0f, palpha = 3.0f, pbeta = 0.75f;
    engine engine;
    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ pb, pf, px, py } });
    set_values(input, {
        -1.0f, -2.0f, -1.3f, -2.2f, -1.1f, -3.5f, -2.0f, -2.0f,
        -0.5f, -1.7f, -5.5f, -1.7f, -0.5f, -1.7f, -0.5f, -6.2f,
        0.2f, -1.2f,  0.0f, -1.2f, 0.0f, -1.2f,  0.0f, -1.2f,
        0.7f, -0.7f, 0.5f, -0.7f, 0.5f, -0.7f, 0.5f, -0.7f,
        2.3f, -5.2f, 1.4f,  2.2f, 1.9f, -6.2f, 1.0f, -0.2f,
        1.4f,  0.3f, 1.5f,  0.3f, 1.5f,  0.3f, 1.5f,  0.3f,
        2.0f,  0.8f, 2.0f,  2.8f, 2.0f,  1.8f, 2.0f,  3.4f,
        2.0f,  0.8f, 3.4f,  0.8f, 2.0f,  0.8f, 5.0f,  6.8f,

        0.1f,   4.2f, 0.1f,  2.0f, 3.1f, 5.0f, 0.1f,  -1.0f,
        2.4f,  -8.0f, 0.4f,  -8.0f, 1.4f,  -8.0f, 0.4f,  -8.0f,
        0.9f,  -1.5f, 2.9f,  -1.5f, 0.9f,  -7.5f, 0.9f,  -1.5f,
        1.4f,  -7.0f, 1.4f,  -1.2f, 1.8f,  -1.0f, 2.4f,  -1.0f,
        1.9f,  -8.5f, 1.9f,  -2.5f, 2.9f,  -6.5f, 3.9f,  -6.5f,
        2.4f,  -1.0f, 2.4f,  -6.0f, 2.4f,  -6.0f, 2.4f,  -6.0f,
        2.9f,  -2.4f, 2.9f,  -5.5f, 4.3f,  -5.9f, 2.9f,  10.5f,
        2.9f,  -5.5f, 2.9f,  -5.5f, 2.9f,  -2.5f, 2.9f,  -8.3f
    });
    VF<float> output_vec = {
        -0.544331f, -0.424837f, -0.0945476f, -0.433174f, -0.560004f, -0.434651f, -0.576648f, -0.118199f, -0.268592f, -0.318447f, -0.400009f, -0.29851f, -0.254547f, -0.198009f, -0.144162f, -0.357563f, 0.129782f, -0.32025f, 0.0f, -0.32025f, 0.0f, -0.32025f, 0.0f, -0.0735643f, 0.165867f, -0.0546491f, 0.208493f, -0.150412f, 0.152754f, -0.0429125f, 0.272166f, -0.309407f, 0.452475f, -0.420244f, 0.391953f, 0.545471f, 0.436366f, -0.389658f, 0.323661f, -0.139282f, 0.213809f, 0.02415f, 0.283724f, 0.0419682f, 0.250737f, 0.0179377f, 0.308142f, 0.0446195f, 0.332026f, 0.418821f, 0.221431f, 0.514605f, 0.325586f, 0.540761f, 0.147786f, 0.159937f, 0.3849f, 0.43116f, 0.414175f, 0.148076f, 0.3849f, 0.243655f, 0.390058f, 0.32024f, 0.0238264f, 0.153235f, 0.0888916f, 0.0835398f, 0.464366f, 0.171115f, 0.0888916f, -0.0431859f, 0.525363f, -0.286054f, 0.0734574f, -0.326213f, 0.200118f, -0.190225f, 0.239641f, -0.336909f, 0.165929f, -0.0423688f, 0.463475f, -0.0628668f, 0.208907f, -0.203997f, 0.194137f, -0.0631705f, 0.31267f, -0.187811f, 0.183863f, -0.199489f, 0.256147f, -0.0315047f, 0.230169f, -0.0561578f, 0.288756f, -0.229789f, 0.288756f, -0.144638f, 0.326293f, -0.242426f, 0.32274f, -0.242426f, 0.266035f, -0.0373802f, 0.266035f, -0.239021f, 0.171742f, -0.17191f, 0.185469f, -0.117475f, 0.271013f, -0.156779f, 0.271013f, -0.177259f, 0.307704f, -0.224662f, 0.271013f, 0.186268f, 0.334362f, -0.366542f, 0.334362f, -0.250442f, 0.238888f, -0.151369f, 0.334362f, -0.168799f
    };

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 2, 0 }, 0 })),
        lrn("lrn", "reorder", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_across_channel, { { 0, 0, 0, 0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);


    // ------------------------------------------------------------------------------------------------
    // test run
    auto outputs = network.execute();
    auto output_memory = outputs.at("lrn").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    try {
        int y_size = output_layout.size.sizes()[3];
        int x_size = output_layout.size.sizes()[2];
        int f_size = output_layout.size.sizes()[1];
        int b_size = output_layout.size.sizes()[0];
        EXPECT_EQ(y_size, py);
        EXPECT_EQ(x_size, px);
        EXPECT_EQ(f_size, pf);
        EXPECT_EQ(b_size, pb);

        for (size_t i = 0; i <output_vec.size(); ++i) {
            EXPECT_NEAR(output_vec[i], output_ptr[i], 1e-04F);

        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }

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

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { pb, pf, px, py } });
    auto output_oracle = memory::allocate(engine, input.get_layout());

    set_values(input, input_oracle_init);
    set_values(output_oracle, output_oracle_init);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(lrn("lrn", "input", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_across_channel));

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

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { pb, pf, px, py } });
    auto output_oracle = memory::allocate(engine, input.get_layout());

    set_values(input, input_oracle_init);
    set_values(output_oracle, output_oracle_init);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(lrn("lrn", "input", psize, pk, palpha, pbeta, cldnn_lrn_norm_region_within_channel));

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

    static std::vector<cldnn::primitive*> generate_specific_test_params()
    {
        std::vector<cldnn_lrn_norm_region> norm_regions = { cldnn_lrn_norm_region_across_channel, cldnn_lrn_norm_region_within_channel };

        //The test checks only valid combinations.
        for (auto norm_region : norm_regions)
        {
            // No padding
            all_layer_params.push_back(new lrn("lrn", "input0", 3, 1.f, 5e-05f, 0.75f, norm_region));
            all_layer_params.push_back(new lrn("lrn", "input0", 5, 17.19f, 0.079f, 0.19f, norm_region));
            
            // Output padding
            all_layer_params.push_back(new lrn("lrn", "input0", 3, 1.f, 5e-05f, 0.75f, norm_region, { { 0, 0, 6, 13 }, 0 }));
            all_layer_params.push_back(new lrn("lrn", "input0", 5, 17.19f, 0.079f, 0.19f, norm_region, { { 0, 0, 11, 5 },{ 0, 0, 0, 19 } }));

            // Input padding
            all_layer_params.push_back(new lrn("lrn", "reorder0", 3, 1.f, 5e-05f, 0.75f, norm_region));
            all_layer_params.push_back(new lrn("lrn", "reorder0", 5, 17.19f, 0.079f, 0.19f, norm_region));

            // Input + Output padding
            all_layer_params.push_back(new lrn("lrn", "reorder0", 3, 1.f, 5e-05f, 0.75f, norm_region, { { 0, 0, 17, 2 }, 0 }));
            all_layer_params.push_back(new lrn("lrn", "reorder0", 5, 17.19f, 0.079f, 0.19f, norm_region, { { 0, 0, 1, 3 },{ 0, 0, 9, 6 } }));
        }
        
        return all_layer_params;
    }

    static std::vector<tests::test_params*> generate_generic_test_params()
    {
        return generic_test::generate_generic_test_params(all_generic_params);
    }

    virtual bool is_format_supported(cldnn::format format)
    {
        return ((format == cldnn_format_type::cldnn_format_yxfb) || (format == cldnn_format_type::cldnn_format_bfyx) || (format == cldnn_format_type::cldnn_format_byxf));
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
    {
        const cldnn::lrn* lrn = (cldnn::lrn*)layer_params;

        //Output is bfyx
        data_types dt = inputs[0].get_layout().data_type;
        auto output = memory::allocate(engine, cldnn::layout(dt, cldnn::format::bfyx, inputs[0].get_layout().size, lrn->output_padding));

        Type beta = lrn->beta;
        Type k = lrn->k;
        uint32_t size = lrn->size;
        cldnn_lrn_norm_region lrn_norm_region = lrn->norm_region;
        Type alpha_sign = std::signbit(lrn->alpha) ? -1.0f : 1.0f;
        Type alpha = (dt == cldnn::data_types::f32) ? (Type)lrn->alpha : alpha_sign;
        Type alpha_div_by_size = (dt == cldnn::data_types::f32) ? (Type)(lrn->alpha / lrn->size) : alpha_sign;
        Type alpha_div_by_size_abs_sqrt = (dt == cldnn::data_types::f32) ? 1.0f : std::sqrt(std::abs(lrn->alpha / lrn->size));
        Type alpha_abs_sqrt = (dt == cldnn::data_types::f32) ? 1.0f : std::sqrt(std::abs(lrn->alpha));

        auto input_mem = inputs[0].pointer<Type>();
        auto output_mem = output.pointer<Type>();
        int batch = inputs[0].get_layout().size.batch[0];
        int feature = inputs[0].get_layout().size.feature[0];
        int height = inputs[0].get_layout().size.spatial[1];
        int width = inputs[0].get_layout().size.spatial[0];

        int output_height = output.get_layout().get_buffer_size().spatial[1];
        int output_width = output.get_layout().get_buffer_size().spatial[0];

        //Initialized output with zeros.
        std::fill(output_mem.begin(), output_mem.end(), static_cast<Type>(0));

        const auto input_desc = get_linear_memory_desc(inputs[0].get_layout());

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
                                    size_t input_index = get_linear_index(inputs[0].get_layout(), n, i, h, w, input_desc);
                                    Type value = input_mem[input_index] * alpha_div_by_size_abs_sqrt;
                                    scale += value * value;
                                }
                                scale = scale * alpha_div_by_size + k;

                                int output_index = (n * feature + c) * output_height * output_width;
                                tensor lower_padding = lrn->output_padding.lower_size(); 
                                output_index += (lower_padding.spatial[1] + h) * output_width + lower_padding.spatial[0] + w;

                                size_t input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w, input_desc);
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
                                        size_t input_index = get_linear_index(inputs[0].get_layout(), n, c, nh, nw, input_desc);
                                        Type value = input_mem[input_index] * alpha_abs_sqrt;
                                        scale += value * value;
                                    }
                                }
                                scale /= pool_size;
                                size_t input_index = get_linear_index(inputs[0].get_layout(), n, c, h, w, input_desc);

                                int output_index = (n * feature + c) * output_height * output_width;
                                tensor lower_padding = lrn->output_padding.lower_size();
                                output_index += (lower_padding.spatial[1] + h) * output_width + lower_padding.spatial[0] + w;

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

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<tests::test_params*, cldnn::primitive*>>& info)
    {
        std::stringstream res;

        const auto & p = std::get<0>(info.param);
        const auto & v = std::get<1>(info.param);

        assert (p->data_type == data_types::f32 ||
                p->data_type == data_types::f16);

        res << info.index
            << "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < p->input_layouts.size(); ++i)
        {
            const auto chans = format::traits(p->fmt).order;

            res << "_" << "Input" << i;
            for (unsigned int j = 0; j < p->input_layouts[i].size.sizes(p->fmt).size(); ++j)
            {
                res << chans[j] << p->input_layouts[i].size.sizes()[j];
            }
        }

        const auto layer = static_cast<cldnn::lrn *>(v);
        res << (layer->norm_region == cldnn_lrn_norm_region_across_channel ? "_Across" : "_Within")
            << "_Size" << layer->size;

        // TODO: the following values need ot be escaped into an acceptable fmt
//            << "_Alpha"  // << layer->alpha
//            << "_Beta"   // << layer->beta
//            << "_K" << layer->k;

        return res.str();
    }

private:

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<cldnn::primitive*> all_layer_params;
    
};

std::vector<cldnn::primitive*> lrn_test::all_layer_params = {};
std::vector<tests::test_params*> lrn_test::all_generic_params = {};

TEST_P(lrn_test, LRN)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(DISABLED_LRN, 
                        lrn_test, 
                        ::testing::Combine(::testing::ValuesIn(lrn_test::generate_generic_test_params()),
                                           ::testing::ValuesIn(lrn_test::generate_specific_test_params())), 
                        lrn_test::custom_param_name);

