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

#include "tests/gtest/gtest.h"
#include "tests/test_utils/test_common_tools.h"
#include "api/neural.h"

using namespace neural;
using namespace std;

class softmax_xb_f32_test_fixture: public ::testing::Test {
public:
    static const uint32_t output_x  = 10, output_b  = 2,  // size of whole output buffer
                   input_x   = 10, input_b   = 2,  // size of whole input buffer
                   in_size  = input_x*input_b,
                   out_size  = output_x*output_b;


    float in_buffer[in_size];
    float out_buffer[out_size];
    float expected_buffer[out_size];
    // input buffer should be initialized with valid data

    neural::primitive input  = memory::create({engine::cpu, memory::format::xb_f32, {input_b, {{input_x}}, 1}});
    neural::primitive output = memory::create({engine::cpu, memory::format::xb_f32, {output_b, {{output_x}}, 1}});

    neural::primitive act    = normalization::softmax::create( {engine::reference,
                                                   output,
                                                   input,
                                                  });

};

TEST_F(softmax_xb_f32_test_fixture, input_same_values) {
    using namespace neural;

// in_buffer filled with same value == 1
    for(size_t i = 0; i < out_size; ++i) {
        in_buffer[i] = 1;
        out_buffer[i] = 0;
        expected_buffer[i] = 0.1f;
    }

    execute({input(in_buffer), output(out_buffer), act});

    bool result = true;
    float sum_result = 0;
    for(size_t i = 0; i < out_size; ++i) {
        result = result && (are_equal(out_buffer[i], expected_buffer[i]));
        sum_result += out_buffer[i];
    }
    EXPECT_EQ(true, result);

}

TEST_F(softmax_xb_f32_test_fixture, input_same_values_batch_wise) {
    using namespace neural;

// in_buffer filled with same value == 1..2 batch wise (softmax can only xb_f32 )
    for(size_t i = 0; i < output_x; ++i) {
        for(size_t j = 0; j < output_b; ++j)
            in_buffer[j+i*output_b] = (j+i*output_b) % 2 +1.0f;
    }

    // fill with the expected
    for(size_t i = 0; i < out_size; ++i) {
        out_buffer[i] = 0.0f;
        expected_buffer[i] = 0.1f;
    }

    execute({input(in_buffer), output(out_buffer), act});

    // does output have expected values
    bool result = true;
    float sum_result = 0;
    for(size_t i = 0; i < out_size; ++i)
        result = result && (are_equal(out_buffer[i], expected_buffer[i]));
    EXPECT_EQ(true, result);

    // does it sum to 1 batch wise ?
    float one = 1.0f;
    for(size_t j = 0; j < output_b; ++j) {
        sum_result = 0;
        for(size_t i = 0; i < output_x; ++i) {
            sum_result += out_buffer[j+i*output_b];
        }

        EXPECT_EQ(true, are_equal(sum_result,one));
    }

}


TEST_F(softmax_xb_f32_test_fixture, values_batch_wise) {
    using namespace neural;

    float in_buffer[in_size] = {
       //b0  b1
        2.0f, 2.0f, //x0
        2.0f, 2.0f, //x1
        2.0f, 2.0f, //x2
        3.0f, 3.0f, //x3
        5.0f, 5.0f, //x4
        4.0f, 4.0f, //x5
        3.0f, 3.0f, //x6
        2.0f, 2.0f, //x7
        2.0f, 2.0f, //x8
        2.0f, 2.0f  //x9
    };

    float expected_buffer[out_size] = {
        0.02569957f,	 0.02569957f,
        0.02569957f,	 0.02569957f,
        0.02569957f,	 0.02569957f,
        0.069858674f,    0.069858674f,
        0.516189665f,    0.516189665f,
        0.189895565f,    0.189895565f,
        0.069858674f,    0.069858674f,
        0.02569957f,	 0.02569957f,
        0.02569957f,	 0.02569957f,
        0.02569957f,	 0.02569957f

    };

    // clean the out_buffer
    for(size_t i = 0; i < out_size; ++i) {
        out_buffer[i] = 0.0f;
    }

    execute({input(in_buffer), output(out_buffer), act});

    // does output have expected values
    bool result = true;
    float sum_result = 0;
    for(size_t i = 0; i < out_size; ++i)
        result = result && (are_equal(out_buffer[i], expected_buffer[i]));
    EXPECT_EQ(true, result);

    // does it sum to 1 batch wise ?
    float one = 1.0f;
    for(size_t j = 0; j < output_b; ++j) {
        sum_result = 0;
        for(size_t i = 0; i < output_x; ++i) {
            sum_result += out_buffer[j+i*output_b];
        }

        EXPECT_EQ(true, are_equal(sum_result,one));
    }

}


TEST(softmax_xb_f32_test, basic_with_offsets) {
    using namespace neural;

    const uint32_t output_x  = 7, output_b  = 3,  // size of whole output buffer
                   input_x   = 6, input_b   = 2,  // size of whole input buffer
                   out_off_x = 0, out_off_b = 1,
                   out_siz_x = 5, out_siz_b = 2;  // size of area to do softmax after offset

    const int32_t  in_off_x  = 1, in_off_b  = 0;

    float in_buffer[input_x*input_b];
    float out_buffer[output_x*output_b];
    // input buffer should be initialized with valid data

    auto input  = memory::create({engine::cpu, memory::format::xb_f32, {input_b, {{input_x}}, 1}});
    auto output = memory::create({engine::cpu, memory::format::xb_f32, {output_b, {{output_x}}, 1}});

    auto act    = normalization::softmax::create( {engine::reference,
                                                   output,
                                                   {out_off_b, {{out_off_x}}, 0},
                                                   {out_siz_b, {{out_siz_x}}, 1},
                                                   input,
                                                   {in_off_b, {{in_off_x}}, 0}
                                                  });

    for(size_t i = 0; i < input_x*input_b; ++i)
        in_buffer[i] = 1;

    float just_a_value = -1.0f;
    for(size_t i = 0; i < output_x*output_b; ++i)
        out_buffer[i] = just_a_value;

    execute({input(in_buffer), output(out_buffer), act});

    bool result = true;
    float expected_value = 0.2f;

    result = true;
    for(size_t i = 0; i < output_x; ++i)
        for(size_t j = 0; j < output_b; ++j) {
            float value = out_buffer[j+i*output_b];
            if((j >= out_off_b && j < (out_off_b+out_siz_b)) && (i >= out_off_x && i < (out_off_x+out_siz_x)))
                result = result && are_equal(value,expected_value); // positions concerning offsets and output size
            else
                result = result && are_equal(value,just_a_value); // skipped positions (bof offsets etc.)
        }

    EXPECT_EQ(true, result);
};
