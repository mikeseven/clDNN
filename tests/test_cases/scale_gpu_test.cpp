/*
// Copyright (c) 2017 Intel Corporation
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
#include "api/primitives/scale.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(scale_gpu, basic_in2x3x2x2_scale_same_size) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25   b1:   0.3   0.4   0.5
    //  f0: b0:  0.6    0.7  0.75   b1:   0.8   0.9   1  
    //  f1: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5     
    //  f1: b0:  1.6    1.7  1.75   b1:   1.8   1.9   2

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 2 } } });
    auto scale_input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 2 } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", false));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.3f, 1.1f, 1.3f,
        0.2f, 0.4f, 1.2f, 1.4f,
        0.25f, 0.5f, 1.25f, 1.5f,
        0.6f, 0.8f, 1.6f, 1.8f,
        0.7f, 0.9f, 1.7f, 1.9f,
        0.75f, 1.f, 1.75f, 2.f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_input_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_same_size_bfyx) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25   b1:   0.3   0.4   0.5
    //  f0: b0:  0.6    0.7  0.75   b1:   0.8   0.9   1  
    //  f1: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5     
    //  f1: b0:  1.6    1.7  1.75   b1:   1.8   1.9   2

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 2, 2, 2, 3 } } });
    auto scale_input = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ 2, 2, 2, 3 } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", false));

    std::vector<float> input_vec = {
        1.f, 2.f, -10.f, 0.f, 0.f, -11.f,
        3.f, 4.f, -14.f, 0.5f, -0.5f, -15.f,
        5.f, 6.f, -12.f, 1.5f, 5.2f, -13.f,
        7.f, 8.f, -16.f, 12.f, 8.f, -17.f
    };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.75f, 0.8f, 0.9f, 1.f,
        1.1f, 1.2f, 1.25f, 1.3f, 1.4f, 1.5f,
        1.6f, 1.7f, 1.75f, 1.8f, 1.9f, 2.f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_input_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_same_size_scale_bfyx) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25   b1:   0.3   0.4   0.5
    //  f0: b0:  0.6    0.7  0.75   b1:   0.8   0.9   1  
    //  f1: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5     
    //  f1: b0:  1.6    1.7  1.75   b1:   1.8   1.9   2

    engine engine;

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ y_size, x_size, feature_num, batch_num } } });
    auto scale_input = memory::allocate(engine, { data_types::f32,{ format::bfyx,{ batch_num, feature_num, y_size, x_size } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", false));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.75f, 0.8f, 0.9f, 1.f,
        1.1f, 1.2f, 1.25f, 1.3f, 1.4f, 1.5f,
        1.6f, 1.7f, 1.75f, 1.8f, 1.9f, 2.f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = l + x_size * (k + y_size * (j + i * feature_num));
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_same_size_bias_term) {
    //  Scale  : 2x3x2x2
    //  Bias   : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25   b1:   0.3   0.4   0.5
    //  f0: b0:  0.6    0.7  0.75   b1:   0.8   0.9   1  
    //  f1: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5     
    //  f1: b0:  1.6    1.7  1.75   b1:   1.8   1.9   2
    //
    //  Bias:
    //  f0: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5
    //  f0: b0:  2.6    2.7  2.75   b1:   2.8   2.9   2  
    //  f1: b0:  3.1    3.2  3.25   b1:   3.3   3.4   3.5     
    //  f1: b0:  4.6    4.7  4.75   b1:   4.8   4.9   4

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 2 } } });
    auto scale_input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 2 } } });
    auto bias = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 2, 3, 2, 2 } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(input_layout("bias", bias.get_layout()));
    topology.add(scale("scale", "input", "scale_input", true, "bias"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.3f, 1.1f, 1.3f,
        0.2f, 0.4f, 1.2f, 1.4f,
        0.25f, 0.5f, 1.25f, 1.5f,
        0.6f, 0.8f, 1.6f, 1.8f,
        0.7f, 0.9f, 1.7f, 1.9f,
        0.75f, 1.f, 1.75f, 2.f
    };
    set_values(scale_input, scale_input_vec);

    std::vector<float> bias_vec = {
        1.1f, 2.3f, 3.1f, 4.3f,
        1.2f, 2.4f, 3.2f, 4.4f,
        1.25f, 2.5f, 3.25f, 4.5f,
        1.6f, 2.8f, 3.6f, 4.8f,
        1.7f, 2.9f, 3.7f, 4.9f,
        1.75f, 2.f, 3.75f, 4.f
    };
    set_values(bias, bias_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);
    network.set_input_data("bias", bias);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_input_vec[i] + bias_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_scalar) {
    //  Scale  : 1
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  0.1    0.2

    engine engine;

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ y_size, x_size, feature_num, batch_num } } });
    auto scale_input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 1, 1, 1, 1 } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", false));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f,
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = 0;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_y) {
    //  Scale  : 2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  0.1    0.2

    engine engine;

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ y_size, x_size, feature_num, batch_num } } });
    auto scale_input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ y_size, 1, 1, 1 } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", false));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f,
        0.2f,
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = k;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_x) {
    //  Scale  : 3
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  0.1    0.2  0.25

    engine engine;

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ y_size, x_size, feature_num, batch_num } } });
    auto scale_input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ 1, x_size, 1, 1 } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", false));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f,
        0.2f,
        0.25f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = l;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_xy) {
    //  Scale  : 2x3x1x1
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0:  0.1    0.2  0.25
    //  f0:  0.6    0.7  0.75

    engine engine;

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ y_size, x_size, feature_num, batch_num } } });
    auto scale_input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ y_size, x_size, 1, 1 } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", false));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f,
        0.2f,
        0.25f,
        0.6f,
        0.7f,
        0.75f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = l + x_size * k;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_batch1) {
    //  Scale  : 2x3x2x1
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25
    //  f0: b0:  0.6    0.7  0.75
    //  f1: b0:  1.1    1.2  1.25    
    //  f1: b0:  1.6    1.7  1.75

    engine engine;

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ y_size, x_size, feature_num, batch_num } } });
    auto scale_input = memory::allocate(engine, { data_types::f32,{ format::yxfb,{ y_size, x_size, feature_num, 1 } } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", false));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 1.1f,
        0.2f, 1.2f,
        0.25f, 1.25f,
        0.6f, 1.6f,
        0.7f, 1.7f,
        0.75f, 1.75f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = j + feature_num * (l + x_size * k);
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Negative Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

//TODO: this should be done using TEST_P or some equivallent construct
static network setup_scale_network(
    const data_types dt,
    const tensor input_tensor,
    const tensor scale_tensor,
    bool bias_term,
    const tensor bias_tensor,
    bool pass_bias_term,    //TODO: a WA for lack of std::optional<bool> bias_term
    bool pass_bias          //TODO: a WA for lack of std::optional<tensor> bias
)
{
    engine engine;
    topology topology;

    auto input_mem = memory::allocate(engine, { dt, input_tensor });
    auto scale_mem = memory::allocate(engine, { dt, scale_tensor });
    topology.add(input_layout("input", input_mem.get_layout()));
    topology.add(input_layout("scale_input", scale_mem.get_layout()));

    if (pass_bias_term)
    {
        if (pass_bias)
        {
            auto bias_mem = memory::allocate(engine, { dt, bias_tensor });
            topology.add(input_layout("bias_input", bias_mem.get_layout()));

            topology.add(scale("scale", "input", "scale_input", bias_term, "bias_input" ));
        }
        else
        {
            topology.add(scale("scale", "input", "scale_input", bias_term));
        }
    }
//TODO: this will be supported after the API change
//    else
//    {
//        assert(!pass_bias);
//
//        topology.add(scale("scale", "input", "scale_input"));
//    }

    return network(engine, topology);
}

TEST(NegativeScaleTest, TestAll) {
    auto d = data_types::f32;
    auto f = format::bfyx;
    auto of = format::yxfb;

    std::vector<int> t { 3, 4, 5, 6 };

    // broadcast rules mean that either the dim size is equal to input dim or is 1
    std::vector<std::vector<int>> good_ts =
    {
        { 1, 4, 5, 6 }, { 3, 1, 5, 6 }, { 3, 4, 1, 6 }, { 3, 4, 5, 1 },
        { 1, 1, 5, 6 }, { 1, 4, 1, 6 }, { 1, 4, 5, 1 }, { 3, 1, 1, 6 }, { 3, 1, 5, 1 }, { 3, 4, 1, 1 },
        { 1, 1, 1, 6 }, { 1, 1, 5, 1 }, { 1, 4, 1, 1 }, { 3, 1, 1, 1 }
    };
    std::vector<std::vector<int>> bad_ts = { { 2, 4, 5, 6 }, { 3, 2, 5, 6 }, { 3, 4, 2, 6 }, { 3, 4, 5, 2 } };

    //TODO: should be ASSERT_THROW(statement, exception_type) - but what exception type?
    ASSERT_ANY_THROW(setup_scale_network(d, { }, { }, false, { }, false, false));
    ASSERT_ANY_THROW(setup_scale_network(d, { }, { }, false, { }, true, false));
    ASSERT_ANY_THROW(setup_scale_network(d, { }, { }, false, { }, true, true));

    ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, t), tensor(of, t), false, tensor(f, t), true, false));
    ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, t), tensor(of, t), true, tensor(f, t), true, true));
    ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, t), tensor(f, t), true, tensor(of, t), true, true));

    // make sure that it's the input that's masked in the scale/bias with a "1", not ther other way around
    for (const auto & good : good_ts)
    {
        ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, good), tensor(f, t), false, tensor(f, t), true, false));
        ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, good), tensor(f, t), true, tensor(f, t), true, true));
    }

    // sizes must either be equal to input or at most have 
    for (const auto & bad : bad_ts)
    {
        ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, t), tensor(f, bad), false, tensor(f, t), true, false));
        ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, t), tensor(f, bad), true, tensor(f, t), true, true));
        ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, t), tensor(f, t), true, tensor(f, bad), true, true));

        for (const auto & good : good_ts)
        {
            ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, t), tensor(f, bad), true, tensor(f, good), true, true));
            ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, t), tensor(f, good), true, tensor(f, bad), true, true));
        }
    }

    // we expect the broadcast mask to be identical for scale and bias, when present
    for (unsigned i = 0; i < good_ts.size(); ++i)
    for (unsigned j = 0; j < good_ts.size(); ++j)
        if (i != j)
            ASSERT_ANY_THROW(setup_scale_network(d, tensor(f, t), tensor(f, good_ts[i]), true, tensor(f, good_ts[j]), true, true));

}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Positive Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

using namespace cldnn;

namespace {
    struct extraScaleTestParam {
        bool bias_term;
    };
}

class scale_test : public tests::generic_test
{
public:
    static void TearDownTestCase()
    {
        for (auto & p : all_generic_params)
            if (p->opaque_custom_param)
                delete reinterpret_cast<extraScaleTestParam *>(p->opaque_custom_param);

        all_generic_params.clear();
        all_layer_params.clear();
    }

    virtual void print_params() override
    {
        const auto p = reinterpret_cast<cldnn::scale *>(layer_params);

        printf("Layer params: bias_term: %d\n", p->bias_term);
    }

    //TODO: use an enum instead of int i
    static std::vector<cldnn::primitive*> generate_specific_test_params(int variant)
    {
        std::vector<cldnn::primitive*> all_layer_params;

        switch(variant)
        {
            case 0: all_layer_params.push_back(new scale("scale", "input0", "input1", false)); break;	//TODO: remove the false here!
            case 1: all_layer_params.push_back(new scale("scale", "input0", "input1", true, "input2")); break;
            case 2: all_layer_params.push_back(new scale("scale", "input0", "input1", false, "input2")); break;
                    //	case 3: all_layer_params.push_back(new scale("scale", "input0", "input1", true));	// This case should be checked by negative_scale_test
                    //	case 4: all_layer_params.push_back(new scale("scale", "input0", "input1", false));	// This case should be checked by negative_scale_test
            default: assert(0);
        }

        return all_layer_params;
    }

    static std::vector<tests::test_params*> generate_generic_test_params(int variant)
    {
        assert(variant >= 0 && variant <= 2);

        std::vector<tests::test_params*> all_generic_params;

        for (cldnn::data_types dt : test_data_types)
        for (cldnn::format fmt : test_formats)
        for (tensor & t : test_input_sizes)
        {
            std::vector<std::vector<int>> attempted_dims;

            for (int32_t b : test_batch_sizes)
            for (auto f : test_feature_sizes)
            for (int mask = 0; mask < 16; ++mask)	//TODO: do we want to restrict it to some smaller subset like for (auto mask : { 0, 1, 3, 7, 15, 5, 10})? the problem is that because of the layout we might miss some interesting combinations since this is effectively hardcoded int he kernels
            {
                const int w = t.spatial[0];
                const int h = t.spatial[1];

                const auto mb = mask & 0x8 ? b : 1;
                const auto mf = mask & 0x4 ? f : 1;
                const auto mh = mask & 0x2 ? h : 1;
                const auto mw = mask & 0x1 ? w : 1;

                // avoid adding test cases with different masks leading to the same dimensions
                if(attempted_dims.end() == std::find_if(attempted_dims.begin(), attempted_dims.end(), [=](const std::vector<int> & arr) { return arr[0] == mb && arr[1] == mf && arr[2] == mh && arr[3] == mw; }))
                {
                    std::vector<int> tmp { mb, mf, mh, mw };
                    attempted_dims.push_back(tmp);

                    test_params * tp = new test_params();
                    tp->data_type = dt;

                    tp->input_layouts.push_back( cldnn::tensor( fmt, { b, f, h, w } ));
                    tp->input_layouts.push_back( cldnn::tensor( fmt, { mb, mf, mh, mw } ));
                    if (variant)
                            tp->input_layouts.push_back( cldnn::tensor( fmt, { mb, mf, mh, mw } ));

                    auto extra_param = new extraScaleTestParam { variant == 1 };
                    tp->opaque_custom_param = extra_param;

                    all_generic_params.emplace_back(tp);
                }
            }
        }

        return all_generic_params;
    }

    static std::vector<std::tuple<test_params*, cldnn::primitive*>> generate_all_test_params()
    {
        std::vector<std::tuple<test_params*, cldnn::primitive*>> res;

        for (int variant = 0; variant <= 2; ++variant)
        {
            auto tpv = generate_generic_test_params(variant); 
            auto pv = generate_specific_test_params(variant);

            for (auto & tp : tpv)
                all_generic_params.emplace_back(tp);

            for (auto & p : pv)
                all_layer_params.emplace_back(p);

            for (auto & tp : tpv)
            for (auto & p: pv)
                res.emplace_back(tp, p);
        }

        return res;
    }

    virtual bool is_format_supported(cldnn::format format) override
    {
        return format == cldnn_format_type::cldnn_format_bfyx;
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<memory> & inputs)
    {
        assert(inputs.size() == 3 || inputs.size() == 2);
        const bool bias_input_present = inputs.size() == 3;

        const memory & input = inputs[0];
        const memory & scale = inputs[1];
        const memory * bias = bias_input_present ? &inputs[2] : nullptr;
        assert(!bias_input_present || bias);

        //Output is bfyx
        auto output = memory::allocate(engine, cldnn::layout(input.get_layout().data_type, input.get_layout().size.transform(cldnn::format::bfyx, 0)));

        const auto params = reinterpret_cast<cldnn::scale *>(layer_params);
        const bool bias_term = params->bias_term;

        const Type * const in0_mem = input.pointer<Type>().data();
        const Type * const in1_mem = scale.pointer<Type>().data();
        const Type * const in2_mem = bias && bias_term ? bias->pointer<Type>().data() : nullptr; //TODO: is the condition needed or is it nullptr anyway?
        Type * const out_mem = output.pointer<Type>().data();

        const int in0_b = input.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0];
        const int in0_f = input.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1];
        const int in0_h = input.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2];
        const int in0_w = input.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3];

        { // asserting dims
            const int out_b = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0]; (void) out_b;
            const int out_f = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1]; (void) out_f;
            const int out_h = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2]; (void) out_h;
            const int out_w = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3]; (void) out_w;

            const int in1_b = scale.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0]; (void) in1_b;
            const int in1_f = scale.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1]; (void) in1_f;
            const int in1_h = scale.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2]; (void) in1_h;
            const int in1_w = scale.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3]; (void) in1_w;
            // input and output dims must match
            assert(in0_b == out_b && in0_f == out_f && in0_h == out_h && in0_w == out_w);

            // input and output dims must match
            assert(in0_b == out_b && in0_f == out_f && in0_h == out_h && in0_w == out_w);

            // scale/bias dims must be equal to in/out or be 1 for broadcast
            assert(in1_b == 1 || in1_b == in0_b);
            assert(in1_f == 1 || in1_f == in0_f);
            assert(in1_h == 1 || in1_h == in0_h);
            assert(in1_w == 1 || in1_w == in0_w);

            if (bias_term)
            {
                const int in2_b = scale.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0]; (void) in2_b;
                const int in2_f = scale.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1]; (void) in2_f;
                const int in2_h = scale.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2]; (void) in2_h;
                const int in2_w = scale.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3]; (void) in2_w;

                // scale and bias dims must match
                assert(in1_b == in2_b && in1_f == in2_f && in1_h == in2_h && in1_w == in2_w);
            }
        }

        for (int n = 0; n < in0_b; ++n)
        for (int c = 0; c < in0_f; ++c)
        for (int y = 0; y < in0_h; ++y)
        for (int x = 0; x < in0_w; ++x)
        {
            const size_t in0_idx = get_linear_index(input.get_layout(), n, c, y, x);
            const size_t in1_idx = get_linear_index_with_broadcast(scale.get_layout(), n, c, y, x, input.get_layout());
            const size_t out_idx = get_linear_index(output.get_layout(), n, c, y, x);

            out_mem[out_idx] = in0_mem[in0_idx] * in1_mem[in1_idx];

            if (bias_term)
            {
                const size_t in2_idx = get_linear_index_with_broadcast(bias->get_layout(), n, c, y, x, input.get_layout());
                out_mem[out_idx] += in2_mem[in2_idx];
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

        const auto & generic_params = std::get<0>(info.param);

        assert(generic_params->opaque_custom_param);
        bool bias_term = reinterpret_cast<extraScaleTestParam *>(generic_params->opaque_custom_param)->bias_term;

        res << info.index
            << "_DT" << (generic_params->data_type == data_types::f32 ? "f32" : "f16")
            << "_InputFMT" << (generic_params->input_layouts[0].format == cldnn::format::bfyx ? "bfyx" : "other")
            << "_InputDims" << generic_params->input_layouts[0].sizes()[0]
                << "x" << generic_params->input_layouts[0].sizes()[1]
                << "x" << generic_params->input_layouts[0].sizes()[2]
                << "x" << generic_params->input_layouts[0].sizes()[3]
            << "_ScaleFMT" << (generic_params->input_layouts[1].format == cldnn::format::bfyx ? "bfyx" : "other")
            << "_ScaleDims" << generic_params->input_layouts[1].sizes()[0]
                << "x" << generic_params->input_layouts[1].sizes()[1]
                << "x" << generic_params->input_layouts[1].sizes()[2]
                << "x" << generic_params->input_layouts[1].sizes()[3]
            << "_BiasTerm" << bias_term;

        if (generic_params->input_layouts.size() > 2)
            res << "_BiasFMT" << (generic_params->input_layouts[2].format == cldnn::format::bfyx ? "bfyx" : "other")
                << "_ScaleDims" << generic_params->input_layouts[2].sizes()[0]
                    << "x" << generic_params->input_layouts[2].sizes()[1]
                    << "x" << generic_params->input_layouts[2].sizes()[2]
                    << "x" << generic_params->input_layouts[2].sizes()[3];

        return res.str();
    }

private:
    static std::vector<std::unique_ptr<tests::test_params>> all_generic_params;
    static std::vector<std::unique_ptr<cldnn::primitive>> all_layer_params;
};

std::vector<std::unique_ptr<cldnn::primitive>> scale_test::all_layer_params = {};
std::vector<std::unique_ptr<tests::test_params>> scale_test::all_generic_params = {};

TEST_P(scale_test, DISABLED_TestAll)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(SCALE,
    scale_test,
    ::testing::ValuesIn(scale_test::generate_all_test_params()),
    scale_test::custom_param_name);

