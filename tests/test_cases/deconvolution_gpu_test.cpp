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
#include "api/primitives/deconvolution.hpp"
#include <api/primitives/data.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"


using namespace cldnn;
using namespace tests;

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 3x3
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  -14    5     2.25 
    //   18    0.75  7.25   
    //   23    42.5  15.5   

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 2, 1, 1 } } });
    auto weights = memory::allocate(engine, { data_types::f32,{ format::yxio,{ 2, 2, 1, 1 } } });
    auto biases = memory::allocate(engine, { data_types::f32,{ format::x,{ 1 } } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { format::yx,{ 0,0 } }, { format::yx,{ 1,1 } })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_nopad_oiyx) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 3x3
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  -14    5     2.25 
    //   18    0.75  7.25   
    //   23    42.5  15.5   

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 2, 1, 1 } } });
    auto weights = memory::allocate(engine, { data_types::f32,{ format::oiyx,{ 1, 1, 2, 2 } } });
    auto biases = memory::allocate(engine, { data_types::f32,{ format::x,{ 1 } } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { format::yx,{ 0,0 } }, { format::yx,{ 1,1 } })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 1x1
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  0.75  

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 2, 1, 1 } } });
    auto weights = memory::allocate(engine, { data_types::f32,{ format::yxio,{ 2, 2, 1, 1 } } });
    auto biases = memory::allocate(engine, { data_types::f32,{ format::x,{ 1 } } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { format::yx,{ 1,1 } }, { format::yx,{ 1,1 } })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(0.75f, output_ptr[0]);
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride2_nopad) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 1x1
    //  Stride : 2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  0.75  

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 2, 1, 1 } } });
    auto weights = memory::allocate(engine, { data_types::f32,{ format::yxio,{ 2, 2, 1, 1 } } });
    auto biases = memory::allocate(engine, { data_types::f32,{ format::x,{ 1 } } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { format::yx,{ 0,0 } }, { format::yx,{ 2,2 } })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -15.f, 5.f, 0.f, 1.25f,
        29.f, 13.f, 2.75f, 1.75,
        -11.f, 4.f, -17.f, 5.5f,
        22.f, 10.f, 32.5f, 14.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride4_pad2) {
    //  Filter : 3x3
    //  Input  : 2x2
    //  Output : 1x1
    //  Stride : 4x4
    //  Pad    : 2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5   1
    //   3.5 1.5   2
    //   3   4     5
    //
    //  Bias
    //  0
    //
    //  Output:
    //  40   0    1.5
    //  0    0    0
    //  6    0   -18

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 2, 1, 1 } } });
    auto weights = memory::allocate(engine, { data_types::f32,{ format::yxio,{ 3, 3, 1, 1 } } });
    auto biases = memory::allocate(engine, { data_types::f32,{ format::x,{ 1 } } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f });
    set_values(biases, { 0.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { format::yx,{ 2,2 } }, { format::yx,{ 4,4 } })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        40.f, 0.f, 1.5f,
        0.f, 0.f, 0.f,
        6.f, 0.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 2, 1, 2 } } });
    auto weights = memory::allocate(engine, { data_types::f32,{ format::yxio,{ 2, 2, 1, 1 } } });
    auto biases = memory::allocate(engine, { data_types::f32,{ format::x,{ 1 } } });

    set_values(input, { 8.f, 1.f, 0.5f, 3.f, 6.f, 2.f, 9.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { format::yx,{ 1,1 } }, { format::yx,{ 2,2 } })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 0.5f, 4.5f, 22.f,
        13.f, 5.f, -17.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2x2_in2x2x1x1_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x1
    //  Output : 2x2x1x1
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  f0:-2   2
    //  f0: 7  -0.5
    //  f1:-2   2
    //  f1: 7  -0.5
    //
    //  Bias
    //  1  5
    //
    //  Output:
    //  f0: -3   4.5 
    //  f0: 13   -17 
    //  f1: 1    8.5
    //  f1: 17 - 13

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 2, 1, 1 } } });
    auto weights = memory::allocate(engine, { data_types::f32,{ format::yxio,{ 2, 2, 1, 2 } } });
    auto biases = memory::allocate(engine, { data_types::f32,{ format::x,{ 2 } } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.f, -2.f, 2.f, 2.f, 7.f, 7.f, -0.5f, -0.5f });
    set_values(biases, { 1.0f, 5.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { format::yx,{ 1,1 } }, { format::yx,{ 2,2 } })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 1.f, 4.5f, 8.5f,
        13.f, 17.f, -17.f, -13.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 2, 1, 2, 2 } } });
    auto weights = memory::allocate(engine, { data_types::f32,{ format::oiyx,{ 1, 1, 2, 2 } } });
    auto biases = memory::allocate(engine, { data_types::f32,{ format::x,{ 1 } } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { format::yx,{ 1,1 } }, { format::yx,{ 2,2 } })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_yxio_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 2, 1, 2, 2 } } });
    auto weights = memory::allocate(engine, { data_types::f32,{ format::yxio,{ 2, 2, 1, 1 } } });
    auto biases = memory::allocate(engine, { data_types::f32,{ format::x,{ 1 } } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { format::yx,{ 1,1 } }, { format::yx,{ 2,2 } })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}