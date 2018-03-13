﻿/*
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

#include <cmath>
#include <gtest/gtest.h>
#include <algorithm>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/activation_grad.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include <api/CPP/data.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(activation_grad_f32_fw_gpu, basic_bfyx_all_functions)
{
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  a: 0.5, b: 2.5
    //

    engine engine;

    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 5, 4 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 5, 4 } });
    auto input_params = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });
    set_values(input_grad,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });

    set_values(input,
    { 12.0f, -22.0f, -32.0f, 42.0f, 52.0f,
        22.0f, 22.0f, 32.0f, 42.0f, -62.0f,
        32.0f, -32.0f, 32.0f, 52.0f, 12.0f,
        12.0f, 12.0f, 12.0f, -12.0f, 12.0f });

    std::vector<cldnn_activation_grad_func> funcs = {
        activation_grad_none,
        activation_grad_relu,
        activation_grad_relu_negative_slope,
    };

    cldnn_activation_additional_params params = { 0.5f, 2.5f };
    set_values(input_params, { params.a, params.b });

    for (uint8_t i = 0; i < 2; i++)
    {
        for (auto func : funcs)
        {
            topology topology(input_layout("input_grad", input_grad.get_layout()));
            topology.add(data("input", input));

            if (i == 0)
            {
                topology.add(activation_grad("activation_grad", "input_grad", "input", func, params));
            }
            else
            {
                topology.add(data("input_params", input_params));
                topology.add(activation_grad("activation_grad", "input_grad", "input", "input_params", func));
            }

            network network(engine, topology);
            network.set_input_data("input_grad", input_grad);
            auto outputs = network.execute();
            EXPECT_EQ(outputs.size(), size_t(1));
            EXPECT_EQ(outputs.begin()->first, "activation_grad");

            auto output_memory = outputs.at("activation_grad").get_memory();
            auto output_layout = output_memory.get_layout();
            auto output_ptr = output_memory.pointer<float>();
            auto input_grad_ptr = input_grad.pointer<float>();
            auto input_ptr = input.pointer<float>();

            int y_size = output_layout.size.spatial[1];
            int x_size = output_layout.size.spatial[0];
            int f_size = output_layout.size.feature[0];
            int b_size = output_layout.size.batch[0];
            EXPECT_EQ(output_layout.format, format::bfyx);
            EXPECT_EQ(y_size, 4);
            EXPECT_EQ(x_size, 5);
            EXPECT_EQ(f_size, 1);
            EXPECT_EQ(b_size, 1);

            std::vector<float> out;

            for (size_t i = 0; i < output_layout.get_linear_size(); ++i)
            {
                switch (func)
                {
                case activation_grad_none:
                    EXPECT_FLOAT_EQ(input_grad_ptr[i], output_ptr[i]);
                    break;
                case activation_grad_relu:
                    EXPECT_FLOAT_EQ(input_grad_ptr[i] * (input_ptr[i] > 0), output_ptr[i]);
                    break;
                case activation_grad_relu_negative_slope:
                        EXPECT_FLOAT_EQ(input_grad_ptr[i] * ((input_ptr[i] > 0) + params.a * (input_ptr[i] <= 0)), output_ptr[i]);
                    break;
                default:
                    break;
                }
            }
        }
    }
}