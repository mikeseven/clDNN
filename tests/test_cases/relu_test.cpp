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

namespace{
    auto calc_idx_obselote = [](std::vector<uint32_t> yxzb_pos, std::vector<uint32_t>& buf_size) -> uint32_t{
        return yxzb_pos[3]
             + yxzb_pos[2] * buf_size[3]
             + yxzb_pos[1] * buf_size[3] * buf_size[2]
             + yxzb_pos[0] * buf_size[3] * buf_size[2] * buf_size[1];
    };
    auto calc_idx = [](std::vector<uint32_t> yxfb_pos, std::vector<uint32_t>& buf_size_bfyx) -> uint32_t{
        return yxfb_pos[3]
             + yxfb_pos[2] * buf_size_bfyx[0]
             + yxfb_pos[1] * buf_size_bfyx[0] * buf_size_bfyx[1]
             + yxfb_pos[0] * buf_size_bfyx[0] * buf_size_bfyx[1] * buf_size_bfyx[3];
    };
}

using namespace neural;

TEST(relu_f32_fw, basic) {
    const uint32_t y = 8, x = 8, f = 3, b = 2;

    auto input  = memory::create({engine::reference, memory::format::yxfb_f32, { b, {y, x}, f}, true});
    auto output = memory::create({engine::reference, memory::format::yxfb_f32, { b, {y, x}, f}});
    input.as<const memory&>().fill<float>();

    auto act = relu::create({engine::reference, output, input});
    auto buf = static_cast<float*>(input.as<const memory&>().pointer);
    // write output to input buffer
    execute({output(buf), act});

    // multiply all positive intigers by -1
    for(size_t i = 0; i < y*x*f*b; ++i)
        buf[i] = (buf[i] > 0)? -buf[i] : buf[i];

    auto act2 = relu::create({engine::reference, output, output});
    execute({act});

    bool result = false;
    // every element should be 0.0f
    for(size_t i = 0; i < y*x*f*b; ++i)
        result = result || buf[i];

    EXPECT_EQ(false, result);
}

TEST(relu_f32_fw, offsets) {
    const uint32_t output_y  = 5,
                   output_x  = 5,
                   output_f  = 2,
                   output_b  = 3, // size of whole output buffer

                   input_y   = 4,
                   input_x   = 5,
                   input_f   = 3,
                   input_b   = 3,  // size of whole input buffer

                   out_off_y = 1,
                   out_off_x = 2,
                   out_off_f = 0,
                   out_off_b = 1,

                   out_siz_y = 3,
                   out_siz_x = 3,
                   out_siz_f = 2,
                   out_siz_b = 2;

     const int32_t in_off_y  = 1,
                   in_off_x  = 1,
                   in_off_f  = 1,
                   in_off_b  = 0;

    vector<uint32_t> in_buf_size  = {input_b , {input_y , input_x }, input_f  };
    vector<uint32_t> out_buf_size = {output_b, {output_y, output_x}, output_f };

    auto input  = memory::create({engine::reference, memory::format::yxfb_f32, in_buf_size, true});
    auto output = memory::create({engine::reference, memory::format::yxfb_f32, out_buf_size, true});
    input.as<const memory&>().fill<float>();

    auto act = relu::create( {engine::reference,
                              output,
                              {out_off_b, {out_off_y, out_off_x}, out_off_f},
                              {out_siz_b, {out_siz_y, out_siz_x}, out_siz_f},
                              input,
                              {in_off_b, {in_off_y, in_off_x}, in_off_f}
                             });

    execute({act});

    auto buf_in  = static_cast<float*>( input.as<const memory&>().pointer);
    auto buf_out = static_cast<float*>(output.as<const memory&>().pointer);
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

TEST(relu_f32_bw, basic) {
    using namespace neural;

    const uint32_t y = 8, x = 8, f = 3, b = 2;

    auto fw_input  = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {y, x, f, b}, true});
    auto bw_input  = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {y, x, f, b}, true});
    auto bw_output = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, {y, x, f, b}, true});
    fw_input.as<const memory_obselote&>().fill<float>();
    bw_input.as<const memory_obselote&>().fill<float>();

    auto act = relu_backward::create({engine::reference, {bw_output}, {bw_input, fw_input}});
    execute({act});

    auto fw_input_buf  = static_cast<float*>(fw_input .as<const memory_obselote&>().pointer);
    auto bw_input_buf  = static_cast<float*>(bw_input .as<const memory_obselote&>().pointer);
    auto bw_output_buf = static_cast<float*>(bw_output.as<const memory_obselote&>().pointer);

    bool result = true;
    for(size_t i = 0; i < y*x*f*b; ++i)
        result &= (((fw_input_buf[i] > 0) * bw_input_buf[i]) == bw_output_buf[i]);

    EXPECT_EQ(true, result);
}

TEST(relu_f32_bw, offsets) {
    using namespace neural;

    const uint32_t output_y  = 7,
                   output_x  = 7,
                   output_f  = 2,
                   output_b  = 3, // size of whole output buffer

                   input_y   = 10,
                   input_x   = 10,
                   input_f   = 3,
                   input_b   = 3,  // size of whole input buffer

                   out_off_y = 1,
                   out_off_x = 2,
                   out_off_f = 0,
                   out_off_b = 1,

                   out_siz_y = 5,
                   out_siz_x = 5,
                   out_siz_f = 2,
                   out_siz_b = 2,

                   fw_in_off_y  = 1,
                   fw_in_off_x  = 1,
                   fw_in_off_f  = 1,
                   fw_in_off_b  = 0,

                   bw_in_off_y  = 0,
                   bw_in_off_x  = 0,
                   bw_in_off_f  = 0,
                   bw_in_off_b  = 1;

    std::vector<uint32_t> fw_in_buf_size  = {input_y, input_x, input_f, input_b};
    std::vector<uint32_t> bw_in_buf_size  = {output_y, output_x, output_f, output_b};
    std::vector<uint32_t> bw_out_buf_size = {output_y, output_x, output_f, output_b};

    auto fw_input  = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, fw_in_buf_size, true});
    auto bw_input  = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, bw_in_buf_size, true});
    auto bw_output = memory_obselote::create({engine::cpu, memory_obselote::format::yxfb_f32, bw_out_buf_size,true});
    fw_input.as<const memory_obselote&>().fill<float>();
    bw_input.as<const memory_obselote&>().fill<float>();

    std::vector<uint32_t> fw_in_off = {fw_in_off_y, fw_in_off_x, fw_in_off_f, fw_in_off_b};
    std::vector<uint32_t> bw_in_off = {bw_in_off_y, bw_in_off_x, bw_in_off_f, bw_in_off_b};

    auto act = relu_backward::create( {engine::reference,
                                      {bw_output},
                                      {out_off_y, out_off_x, out_off_f, out_off_b},
                                      {out_siz_y, out_siz_x, out_siz_f, out_siz_b},
                                      {bw_input, fw_input},
                                      {bw_in_off, fw_in_off}
                                     });
    execute({act});

    auto buf_fw_input  = static_cast<float*>(fw_input.as<const memory_obselote&>().pointer);
    auto buf_bw_input  = static_cast<float*>(bw_input.as<const memory_obselote&>().pointer);
    auto buf_bw_output = static_cast<float*>(bw_output.as<const memory_obselote&>().pointer);

    bool result = true;
    for(uint32_t y = 0; y < out_siz_y; ++y)
    for(uint32_t x = 0; x < out_siz_x; ++x)
    for(uint32_t f = 0; f < out_siz_f; ++f)
    for(uint32_t b = 0; b < out_siz_b; ++b)
    {
        auto fw_in_idx = calc_idx_obselote( {
                                 fw_in_off_y + y,
                                 fw_in_off_x + x,
                                 fw_in_off_f + f,
                                 fw_in_off_b + b
                                }, fw_in_buf_size);

        auto bw_in_idx = calc_idx_obselote( {
                                 bw_in_off_y + y,
                                 bw_in_off_x + x,
                                 bw_in_off_f + f,
                                 bw_in_off_b + b
                                }, bw_in_buf_size);

        auto out_idx = calc_idx_obselote( {
                                 out_off_y + y,
                                 out_off_x + x,
                                 out_off_f + f,
                                 out_off_b + b
                                }, bw_out_buf_size);

        result &= (((buf_fw_input[fw_in_idx] > 0) * buf_bw_input[bw_in_idx]) == buf_bw_output[out_idx]);
    }

    EXPECT_EQ(true, result);
}
