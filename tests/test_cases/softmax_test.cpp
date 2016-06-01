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

#include "api/neural.h"
#include "tests/gtest/gtest.h"
#include "test_utils/test_utils.h"

using namespace neural;
using namespace std;
using namespace tests;

class softmax_xb_f32_test_fixture: public ::testing::Test {
public:
    static const uint32_t
        output_x  = 10, output_b  = 2,  // size of whole output buffer
        input_x   = 10, input_b   = 2,  // size of whole input buffer
        in_size   = input_x*input_b,
        out_size  = output_x*output_b;


    float in_buffer[in_size];
    float out_buffer[out_size];
    float expected_buffer[out_size];

    neural::primitive input  = memory::describe({engine::reference, memory::format::xb_f32, {input_b, {{input_x}}, 1}});
    neural::primitive output = memory::describe({engine::reference, memory::format::xb_f32, {output_b, {{output_x}}, 1}});
    neural::primitive act    = normalization::softmax::create({engine::reference, output, input});

    void compare_out_buffer_with_expected() {
        for(size_t i = 0; i < out_size; ++i) {
            // does output have expected values
            EXPECT_TRUE(are_equal(out_buffer[i], expected_buffer[i]))
                << "At ["<< i <<  "] Expected : " << expected_buffer[i] << " actual : " << out_buffer[i];
        }
    }

    void compare_out_buffer_with_expected_batch_wise() {
        for(size_t b = 0; b < output_b; ++b) {
            float batch_wise_sum = 0;
            for(size_t x = 0; x < output_x; ++x) {
                auto idx = b+x*output_b;
                batch_wise_sum += out_buffer[idx];
                // does output have expected values
                EXPECT_TRUE(are_equal(out_buffer[idx], expected_buffer[idx]))
                    << "At ["<< idx <<  "] Expected : " << expected_buffer[idx] << " actual : " << out_buffer[idx];
            }
            // does it sum to 1 batch wise
            EXPECT_TRUE(are_equal(batch_wise_sum, 1.0f))
                << "Expected : " << 1.0f << " actual : " << batch_wise_sum;
        }
    }
};

TEST_F(softmax_xb_f32_test_fixture, input_same_values) {
// in_buffer filled with same value == 1.0f
    for(size_t i = 0; i < out_size; ++i) {
              in_buffer[i] = 1.0f;
        expected_buffer[i] = 0.1f;
    }
    execute({input(in_buffer), output(out_buffer), act}).wait();
    compare_out_buffer_with_expected();
}

TEST_F(softmax_xb_f32_test_fixture, input_same_values_batch_wise) {
// in_buffer filled with same value == 1..2 each batch accordingly (softmax can only xb_f32 )
    for(size_t i = 0; i < output_x; ++i) {
        for(size_t j = 0; j < output_b; ++j)
            in_buffer[j+i*output_b] = (j+i*output_b) % 2 +1.0f;
    }
    // fill buffer with the expected 0.1f value
    for(size_t i = 0; i < out_size; ++i)
        expected_buffer[i] = 0.1f;

    execute({input(in_buffer), output(out_buffer), act}).wait();
    compare_out_buffer_with_expected_batch_wise();
}


TEST_F(softmax_xb_f32_test_fixture, values_batch_wise) {

    float in_buf[in_size] = {
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

    float exp_buf[out_size] = {
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
    std::copy(in_buf, in_buf+in_size, in_buffer);
    std::copy(exp_buf, exp_buf+in_size, expected_buffer);

    // out_buffer filled with non-signaling NaN
    for(size_t i = 0; i < out_size; ++i)
        out_buffer[i] = NAN;

    execute({input(in_buffer), output(out_buffer), act}).wait();
    compare_out_buffer_with_expected_batch_wise();
}

TEST(softmax_xb_f32_test, basic_with_offsets) {

    const uint32_t output_x  = 7, output_b  = 3,  // size of whole output buffer
                   input_x   = 6, input_b   = 2,  // size of whole input buffer
                   out_off_x = 0, out_off_b = 1,
                   out_siz_x = 5, out_siz_b = 2;  // size of area to do softmax after offset

    const int32_t  in_off_x  = 1, in_off_b  = 0;

    float in_buffer[input_x*input_b];
    float out_buffer[output_x*output_b];
    // input buffer should be initialized with valid data

    auto input  = memory::describe({engine::reference, memory::format::xb_f32, {input_b, {{input_x}}, 1}});
    auto output = memory::describe({engine::reference, memory::format::xb_f32, {output_b, {{output_x}}, 1}});

    auto act    = normalization::softmax::create({engine::reference,
                                                  output,
                                                  {out_off_b, {{out_off_x}}, 0},
                                                  {out_siz_b, {{out_siz_x}}, 1},
                                                  input,
                                                  {in_off_b, {{in_off_x}}, 0}
                                                 });
    // in_buffer filled with same value == 1.0f
    for(size_t i = 0; i < input_x*input_b; ++i)
        in_buffer[i] = 1.0f;

    const float out_of_offset_value = NAN;
    // out_buffer filled with non-signaling NaN
    for(size_t i = 0; i < output_x*output_b; ++i)
        out_buffer[i] = out_of_offset_value;

    execute({input(in_buffer), output(out_buffer), act}).wait();

    auto expected_value = 0.2f;
    auto end_b = out_off_b+out_siz_b;
    auto end_x = out_off_x+out_siz_x;

    for(size_t x = 0; x < output_x; ++x)
        for(size_t b = 0; b < output_b; ++b) {
            auto idx = b+x*output_b;
            float value = out_buffer[idx];
            float expected = (b >= out_off_b && b < end_b) && (x >= out_off_x && x < end_x) //is in range ?
                ? expected_value       // valid value that's in data range
                : out_of_offset_value; // invalid value (non-signaling NaN) for skipped buffer positions (bof offsets)
          EXPECT_TRUE(are_equal(value, expected))
              << "At ["<< idx <<  "] Expected : " << expected << " actual :" << value;
        }

};
