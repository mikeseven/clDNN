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

using namespace neural;
using namespace std;

class Reorder_test_fixture: public ::testing::Test {
public:
    const uint32_t dim_y      = 2, dim_x      = 2, dim_f      = 4, dim_b      = 3;

  	static const uint32_t all_size = 48; //dim_y*dim_x*dim_f*dim_b;
	memory::format::type in_layout = memory::format::yxfb_f32;
    memory::format::type out_layout = memory::format::bfxy_f32;

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
    {//bfxy x=2, y=2, f=4, b=3
        0, // b0 f0 x0 y0
       24, // b0 f0 x0 y1
       12, // b0 f0 x1 y0
       36, // b0 f0 x1 y1
        3, // b0 f1 x0 y0
       27, // b0 f1 x0 y1
       15, // b0 f1 x1 y0
       39, // b0 f1 x1 y1

        6, // b0 f2 x0 y0
       30, // b0 f2 x0 y1
       18, // b0 f2 x1 y0
       42, // b0 f2 x1 y1
        9, // b0 f3 x0 y0
       33, // b0 f3 x0 y1
       21, // b0 f3 x1 y0
       45, // b0 f3 x1 y1

        1, // b1 f0 x0 y0
       25, // b1 f0 x0 y1
       13, // b1 f0 x1 y0
       37, // b1 f0 x1 y1
        4, // b1 f1 x0 y0
       28, // b1 f1 x0 y1
       16, // b1 f1 x1 y0
       40, // b1 f1 x1 y1

        7, // b1 f2 x0 y0
       31, // b1 f2 x0 y1
       19, // b1 f2 x1 y0
       43, // b1 f2 x1 y1
       10, // b1 f3 x0 y0
       34, // b1 f3 x0 y1
       22, // b1 f3 x1 y0
       46, // b1 f3 x1 y1

        //3, // b2 f0 x0 y0 // should fail
        2, // b2 f0 x0 y0
       26, // b2 f0 x0 y1
       14, // b2 f0 x1 y0
       38, // b2 f0 x1 y1
        5, // b2 f1 x0 y0
       29, // b2 f1 x0 y1
       17, // b2 f1 x1 y0
       41, // b2 f1 x1 y1

        8, // b2 f2 x0 y0
       32, // b2 f2 x0 y1
       20, // b2 f2 x1 y0
       44, // b2 f2 x1 y1
       11, // b2 f3 x0 y0
       35, // b2 f3 x0 y1
       23, // b2 f3 x1 y0
       47, // b2 f3 x1 y1
    };

    float wyn_buffer_fail[all_size] =
    {//bfxy x=2, y=2, f=4, b=3
        0,24,12,36,3,27,15,39,6,30,18,42,9,33,21,45,1,25,13,37,4,28,16,40,7,
        31,19,43,10,34,22,46,
        3, // b2 f0 x0 y0 // should fail (2 is OK)
        26,14,38,5,29,17,41,8,32,20,44,11,35,23,47
    };

	// input buffer should be initialized with valid data
                                    //y=2 x=2 f=4 b=3
    std::vector<uint32_t> in_sizes = { dim_y, dim_x, dim_f, dim_b};
                                    //b=3 f=4 x=2 y=2
    std::vector<uint32_t> out_sizes= { dim_b, dim_f, dim_x, dim_y};

    neural::primitive input   = memory::create({engine::cpu, in_layout, in_sizes});
    neural::primitive output  = memory::create({engine::cpu, out_layout, out_sizes, true});
    neural::primitive reorder = reorder::create(reorder::arguments{engine::reference,input,output});
};

TEST_F(Reorder_test_fixture,reorder_test_basic) {

    try
    {
        execute({input(in_buffer), reorder});
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

    auto input2  = memory::create({engine::cpu, out_layout, out_sizes});
    auto output2 = memory::create({engine::cpu, in_layout, in_sizes, true});
    auto reorder2    = reorder::create({engine::reference,input2,output2});

    float* buf_out = nullptr;
    float* buf_out2 = nullptr;
    try
    {
        execute({input(in_buffer), reorder});
        buf_out = static_cast<float*>(output.as<const memory&>().pointer);

        execute({input2(buf_out), reorder2});
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
