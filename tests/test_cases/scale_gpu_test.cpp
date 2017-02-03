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

TEST(scale_gpu, basic_in2x3x2x2_scale_x) {
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

TEST(scale_gpu, basic_in2x3x2x2_scale_y) {
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