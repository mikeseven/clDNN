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

#include "api/neural.h"
#include <gtest/gtest.h>
#include "test_utils/test_utils.h"
#include "memory_utils.h"

namespace{
    auto calc_idx = [](std::vector<uint32_t> yxfb_pos, std::vector<uint32_t>& buf_size_bfyx) -> uint32_t{
        return yxfb_pos[3]
             + yxfb_pos[2] * buf_size_bfyx[0]
             + yxfb_pos[1] * buf_size_bfyx[0] * buf_size_bfyx[1]
             + yxfb_pos[0] * buf_size_bfyx[0] * buf_size_bfyx[1] * buf_size_bfyx[2];
    };
}

using namespace neural;


TEST(relu_f32_fw, basic) {
    const uint32_t y = 8, x = 8, f = 3, b = 2;

    auto input  = memory::allocate({engine::reference, memory::format::yxfb_f32, { b, {y, x}, f}});
    auto& output = input;
    fill<float>(input.as<const memory&>());

    auto act = relu::create({engine::reference, output, input});
    // write output to input buffer
    execute({output, act}).wait();

    // multiply all positive intigers by -1
    auto buf = input.as<const memory&>().pointer<float>();
    for(size_t i = 0; i < y*x*f*b; ++i)
        buf[i] = (buf[i] > 0)? -buf[i] : buf[i];

    execute({act}).wait();

    bool result = false;
    // every element should be 0.0f
    buf = input.as<const memory&>().pointer<float>();
    for(size_t i = 0; i < y*x*f*b; ++i)
        result = result || buf[i];

    EXPECT_EQ(false, result);
}

TEST(relu_f32_fw, DISABLED_intrinsics_avx2) {
    const uint32_t y = 8, x = 8, f = 3, b = 2;

    // Optimized data
    auto input  = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ b,{ y, x }, f }});
    auto output = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ b,{ y, x }, f }});
    auto& input_memory  = input.as<const memory&>();
    auto& output_memory = output.as<const memory&>();

    // Reference data
    auto ref_output = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ b,{ y, x }, f }});
    auto& ref_output_memory = ref_output.as<const memory&>();

    // Initialize input data
    fill<float>(input_memory);

    // Relu primitives
    auto opt_relu = relu::create({ engine::cpu, output, input });
    auto ref_relu = relu::create({ engine::reference, ref_output, input });

    execute({output, opt_relu}).wait();
    execute({ref_output, ref_relu}).wait();

    {
        auto ref_out_ptr = ref_output_memory.pointer<float>();
        auto out_ptr = output_memory.pointer<float>();

        for (size_t output_element = 0; output_element < output_memory.count(); ++output_element)
            EXPECT_EQ(true, tests::are_equal(ref_out_ptr[output_element], out_ptr[output_element]));
    }
}

TEST(relu_f32_fw, offsets) {
    const uint32_t output_x  = 5,
                   output_y  = 5,
                   output_f  = 2,
                   output_b  = 3, // size of whole output buffer

                   input_x   = 4,
                   input_y   = 5,
                   input_f   = 3,
                   input_b   = 3,  // size of whole input buffer

                   out_off_x = 1,
                   out_off_y = 2,
                   out_off_f = 0,
                   out_off_b = 1,

                   out_siz_x = 3,
                   out_siz_y = 3,
                   out_siz_f = 2,
                   out_siz_b = 2;

     const int32_t in_off_x  = 1,
                   in_off_y  = 1,
                   in_off_f  = 1,
                   in_off_b  = 0;

    vector<uint32_t> in_buf_size  = {input_b , { input_x , input_y}, input_f  };
    vector<uint32_t> out_buf_size = {output_b, {output_x, output_y}, output_f };

    auto input  = memory::allocate({engine::reference, memory::format::yxfb_f32, in_buf_size});
    auto output = memory::allocate({engine::reference, memory::format::yxfb_f32, out_buf_size});
    fill<float>(input.as<const memory&>());

    auto act = relu::create( {engine::reference,
                              output,
                              {out_off_b, {out_off_x, out_off_y}, out_off_f},
                              {out_siz_b, {out_siz_x, out_siz_y}, out_siz_f},
                              input,
                              {in_off_b, {in_off_x, in_off_y}, in_off_f}
                             });

    auto buf_in  = input.as<const memory&>().pointer<float>();
    auto buf_out = output.as<const memory&>().pointer<float>();

    execute({act}).wait();

    bool result = true;

    for(uint32_t y = 0; y < out_siz_y; ++y)
    for(uint32_t x = 0; x < out_siz_x; ++x)
    for(uint32_t f = 0; f < out_siz_f; ++f)
    for(uint32_t b = 0; b < out_siz_b; ++b)
    {
        auto in_idx = calc_idx( {
                                 in_off_y + y,
                                 in_off_x + x,
                                 in_off_f + f,
                                 in_off_b + b
                                }, in_buf_size.raw);
        auto out_idx = calc_idx( {
                                 out_off_y + y,
                                 out_off_x + x,
                                 out_off_f + f,
                                 out_off_b + b
                                }, out_buf_size.raw);

        result &= (buf_out[out_idx] > 0.0f)
                  ? (buf_out[out_idx] == buf_in[in_idx])
                  : (buf_in[in_idx] < 0.0f);
    }

    EXPECT_EQ(true, result);
}




