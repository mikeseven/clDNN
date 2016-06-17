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

#include "tests/gtest/gtest.h"
#include "api/neural.h"
#include "memory_utils.h"

using namespace neural;
using namespace std;

class Reorder_test_fixture: public ::testing::Test {
public:
    const uint32_t dim_y      = 2, dim_x      = 2, dim_f      = 4, dim_b      = 3;

  	static const uint32_t all_size = 48; //dim_y*dim_x*dim_f*dim_b;
	memory::format::type in_layout = memory::format::yxfb_f32;
    memory::format::type out_layout = memory::format::byxf_f32; 

    float in_buffer[all_size] =
    {// yxfb
         0,  1,  2,//b row f0 x0 y0
         3,  4,  5,//b row f1 x0 y0
         6,  7,  8,//b row f2 x0 y0
         9, 10, 11,//b row f3 x0 y0

        12, 13, 14,//b row f0 x1 y0
        15, 16, 17,//b row f1 x1 y0
        18, 19, 20,//b row f2 x1 y0
        21, 22, 23,//b row f3 x1 y0

        24, 25, 26,//b row f0 x0 y1
        27, 28, 29,//b row f1 x0 y1
        30, 31, 32,//b row f2 x0 y1
        33, 34, 35,//b row f3 x0 y1

        36, 37, 38,//b row f0 x1 y1
        39, 40, 41,//b row f1 x1 y1
        42, 43, 44,//b row f2 x1 y1
        45, 46, 47 //b row f3 x1 y1
    };

    float wyn_buffer[all_size] =
    {// byxf_f32 x=2, y=2, f=4, b=3
        0, // b0 y0 x0 f0
        3, // b0 y0 x0 f1
        6, // b0 y0 x0 f2
        9, // b0 y0 x0 f3
       12, // b0 y0 x1 f0
       15, // b0 y0 x1 f1
       18, // b0 y0 x1 f2
       21, // b0 y0 x1 f3
       24, // b0 y1 x0 f0
       27, // b0 y1 x0 f1
       30, // b0 y1 x0 f2
       33, // b0 y1 x0 f3
       36, // b0 y1 x1 f0
       39, // b0 y1 x1 f1
       42, // b0 y1 x1 f2
       45, // b0 y1 x1 f3
        1, // b1 y0 x0 f0
        4, // b1 y0 x0 f1
        7, // b1 y0 x0 f2
       10, // b1 y0 x0 f3
       13, // b1 y0 x1 f0
       16, // b1 y0 x1 f1
       19, // b1 y0 x1 f2
       22, // b1 y0 x1 f3
       25, // b1 y1 x0 f0
       28, // b1 y1 x0 f1
       31, // b1 y1 x0 f2
       34, // b1 y1 x0 f3
       37, // b1 y1 x1 f0
       40, // b1 y1 x1 f1
       43, // b1 y1 x1 f2
       46, // b1 y1 x1 f3
        2, // b2 y0 x0 f0
        5, // b2 y0 x0 f1
        8, // b2 y0 x0 f2
       11, // b2 y0 x0 f3
       14, // b2 y0 x1 f0
       17, // b2 y0 x1 f1
       20, // b2 y0 x1 f2
       23, // b2 y0 x1 f3
       26, // b2 y1 x0 f0
       29, // b2 y1 x0 f1
       32, // b2 y1 x0 f2
       35, // b2 y1 x0 f3
       38, // b2 y1 x1 f0
       41, // b2 y1 x1 f1
       44, // b2 y1 x1 f2
       47, // b2 y1 x1 f3
    };

	// input buffer should be initialized with valid data
                                    //y=2 x=2 f=4 b=3
	neural::vector<uint32_t> in_sizes = { dim_b, {dim_y, dim_x}, dim_f };
                                    //b=3 f=4 x=2 y=2
	neural::vector<uint32_t> out_sizes = { dim_b, {dim_y, dim_x}, dim_f };

    neural::primitive input   = memory::describe({engine::reference, in_layout, in_sizes});
    neural::primitive output  = memory::allocate({engine::reference, out_layout, out_sizes});
    neural::primitive reorder = reorder::create({engine::reference,output, input});
};

TEST_F(Reorder_test_fixture,reorder_test_basic) {

    try
    {
        execute({input(in_buffer), reorder}).wait();
    }
    catch (const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }

    auto buf_out = static_cast<float*>(output.as<const memory&>().pointer);

    bool result = true;
    for(size_t i = 0; i < dim_y*dim_x*dim_f*dim_b; ++i)
        result &= buf_out[i] == wyn_buffer[i];

    EXPECT_EQ(true, result);
}

TEST_F(Reorder_test_fixture,reorder_test_output_as_input_2pass) {

    auto input2  = memory::describe({engine::reference, out_layout, out_sizes});
    auto output2 = memory::allocate({engine::reference, in_layout, in_sizes});
    auto reorder2    = reorder::create({engine::reference, output2, input2});

    float* buf_out = nullptr;
    float* buf_out2 = nullptr;
    try
    {
        execute({input(in_buffer), reorder}).wait();
        buf_out = static_cast<float*>(output.as<const memory&>().pointer);

        execute({input2(buf_out), reorder2}).wait();
        buf_out2 = static_cast<float*>(output2.as<const memory&>().pointer);
    }
    catch (const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    bool result = true;

    for(size_t i = 0; i < dim_y*dim_x*dim_f*dim_b; ++i)
        result &= buf_out2[i] == in_buffer[i];

    EXPECT_EQ(true, result);
}

TEST(reorder_test, byxf_f32_to_byxf_b24_f32) {
    const uint32_t y = 10, x = 10, f = 4, b = 24*2;
    auto engine_resource = worker_cpu::create({8});

    auto input      = memory::allocate({engine::reference, memory::format::byxf_f32,     {b, {x, y}, f}});
    auto output     = memory::allocate({engine::reference, memory::format::byxf_b24_f32, {b, {x, y}, f}});
    auto output_ref = memory::allocate({engine::reference, memory::format::byxf_b24_f32, {b, {x, y}, f}});
    fill<float>(input.as<const memory&>());

    auto valid  = reorder::create({engine::reference, output_ref, input});
    auto tested = reorder::create({engine::cpu,       output,     input});

    execute({ valid }).wait();
    auto output_ref_ptr = static_cast<float *>(output_ref.as<const memory&>().pointer);

    execute({ tested }, {engine_resource} ).wait();
    auto output_ptr     = static_cast<float *>(output.as<const memory&>().pointer);

    for(size_t i = 0; i < y*x*f*b; ++i)
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "at index: " << i;
}

TEST(reorder_test, bfyx_f32_to_byxf_f32) {
    const uint32_t y = 5, x = 4, f = 3, b = 12;

    auto input      = memory::allocate({engine::reference, memory::format::bfyx_f32, {b, {x, y}, f}});
    auto output     = memory::allocate({engine::reference, memory::format::byxf_f32, {b, {x, y}, f}});
    auto output_ref = memory::allocate({engine::reference, memory::format::byxf_f32, {b, {x, y}, f}});
    fill<float>(input.as<const memory&>());

    auto valid  = reorder::create({engine::reference, output_ref, input});
    auto tested = reorder::create({engine::cpu,       output,     input});
    execute({valid, tested}).wait();

    auto output_ptr     = static_cast<float *>(output.as<const memory&>().pointer);
    auto output_ref_ptr = static_cast<float *>(output_ref.as<const memory&>().pointer);
    for(size_t i = 0; i < y*x*f*b; ++i)
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]);
}
