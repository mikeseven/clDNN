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
#include "gpu/ocl_toolkit.h"
#include "multidimensional_counter.h"
#include "test_utils/test_utils.h"
#include "memory_utils.h"

#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <limits>

using namespace neural;
using namespace tests;
using namespace testing;

namespace
{
struct gpu_info_helper : gpu::context_holder
{
    gpu::engine_info get_engine_info() const
    {
        return context()->get_engine_info();
    }
};
}

TEST(reorder_gpu_f32, basic_subtract) {
    //  Input               : 2x2x2x2
    //  Output              : 2x2x2x2
    //  Subtract            : 1x2x2x2 (only first batch is taken into consideration)
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Subtract:
    //  f0: b0:  1    1.5
    //  f0: b0:  2    2.5
    //  f1: b0:  4    3
    //  f1: b0:  2    1
    //
    //
    //  Output:
    //  b0 f0:  0    0.5
    //  b0 f0:  1    1.5
    //
    //  b0 f1:  1    3
    //  b0 f1:  5    7
    //
    //  b1 f0: -1   -1.5
    //  b1 f0: -1.5 -3
    //
    //  b1 f1: -2.5  2.2
    //  b1 f1: 10    7
    //

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 }, 2 });
    auto output = memory::allocate({ memory::format::bfyx_f32,{ 2,{ 2, 2 }, 2 }, 2 });
    auto subtract = memory::allocate({ memory::format::byxf_f32,{ 1,{ 2, 2 }, 2 }, 2 });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    set_values(subtract, {
        1.0f,  4.0f,      1.5f,  3.0f,
        2.0f,  2.0f,      2.5f,  1.0f,
    });

    auto reorder = reorder::create({  output, input, subtract });

    execute({ reorder }).wait();

    float answers[16] = { 0.0f,  0.5f,
                          1.0f,  1.5f,

                          1.0f,  3.0f,
                          5.0f,  7.0f,

                         -1.0f, -1.5f,
                         -1.5f, -3.0f,

                         -2.5f,  2.2f,
                         10.0f,  7.0f
    };

    auto output_ptr = output.as<const memory&>().pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reorder_gpu_f32, basic_subtract_value) {
    //  Values_to_subtract  : 2
    //  Input               : 2x2x2x2
    //  Output              : 2x2x2x2
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  subtract values
    //  f0: 0.5
    //  f1: 2.5
    //
    //  Output:
    //  b0 f0:  0.5  1.5
    //  b0 f0:  2.5  3.5
    //
    //  b0 f1:  2.5  3.5
    //  b0 f1:  4.5  5.5
    //
    //  b1 f0: -0.5 -0.5
    //  b1 f0:  0.0 -1.0
    //
    //  b1 f1: -1.0  2.7
    //  b1 f1:  9.5  5.5
    //

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 }, 2 });
    auto output = memory::allocate({ memory::format::bfyx_f32,{ 2,{ 2, 2 }, 2 }, 2 });
    std::vector<float> subtract_val = { 0.5, 2.5 };

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    auto reorder = reorder::create({  output, input, subtract_val, true });

    execute({ reorder }).wait();

    float answers[16] = { 0.5f, 1.5f,
                          2.5f, 3.5f,

                          2.5f, 3.5f,
                          4.5f, 5.5f,

                         -0.5f, -0.5f,
                          0.0f, -1.0f,

                         -1.0f,  2.7f,
                          9.5f,  5.5f
    };

    auto output_ptr = output.as<const memory&>().pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reorder_gpu_f16, basic_subtract_f32_output_f32) {
    //  Input               : 2x2x2x2 (FP16)
    //  Output              : 2x2x2x2 (FP32)
    //  Subtract            : 1x2x2x2 (FP32, only first batch is taken into consideration)
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Subtract (FP32 - converted internally to FP16 before subtraction):
    //  f0: b0:  1    1.5
    //  f0: b0:  2    2.5
    //  f1: b0:  4    3
    //  f1: b0:  2    1
    //
    //
    //  Output:
    //  b0 f0:  0    0.5
    //  b0 f0:  1    1.5
    //
    //  b0 f1:  1    3
    //  b0 f1:  5    7
    //
    //  b1 f0: -1   -1.5
    //  b1 f0: -1.5 -3
    //
    //  b1 f1: -2.5  2.2
    //  b1 f1: 10    7
    //

    gpu_info_helper gpu_info;
    if (!gpu_info.get_engine_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    auto input = memory::allocate({  memory::format::yxfb_f16,{ 2,{ 2, 2 }, 2 }, 2 });
    auto output = memory::allocate({  memory::format::bfyx_f32,{ 2,{ 2, 2 }, 2 }, 2 });
    auto subtract = memory::allocate({  memory::format::byxf_f32,{ 1,{ 2, 2 }, 2 }, 2 });

    set_values(input, {
        half_t(0x3C00), half_t(0x0000), // 1.f, 0.f,
        half_t(0x4500), half_t(0x3E00), // 5.f, 1.5f,

        half_t(0x4000), half_t(0x0000), // 2.f, 0.f,
        half_t(0x4600), half_t(0x4533), // 6.f, 5.2f,

        half_t(0x4200), half_t(0x3800), // 3.f, 0.5f,
        half_t(0x4700), half_t(0x4A00), // 7.f, 12.f,

        half_t(0x4400), half_t(0xB800), // 4.f, -0.5f,
        half_t(0x4800), half_t(0x4800)  // 8.f, 8.f
    });

    set_values(subtract, {
        1.0f,  4.0f,      1.5f,  3.0f,
        2.0f,  2.0f,      2.5f,  1.0f,
    });

    auto reorder = reorder::create({  output, input, subtract });

    execute({ reorder }).wait();

    float answers[16] = { 0.0f,  0.5f,
                          1.0f,  1.5f,

                          1.0f,  3.0f,
                          5.0f,  7.0f,

                         -1.0f, -1.5f,
                         -1.5f, -3.0f,

                         -2.5f,  2.2f,
                         10.0f,  7.0f
    };
    
    auto output_ptr = output.as<const memory&>().pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reorder_gpu_f16, basic_subtract_value) {
    //  Values_to_subtract  : 2
    //  Input               : 2x2x2x2 (FP16)
    //  Output              : 2x2x2x2 (FP16)
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  subtract values (FP32 - converted internally to FP16 before subtraction)
    //  f0: 0.5
    //  f1: 2.5
    //
    //  Output:
    //  b0 f0:  0.5  1.5
    //  b0 f0:  2.5  3.5
    //
    //  b0 f1:  2.5  3.5
    //  b0 f1:  4.5  5.5
    //
    //  b1 f0: -0.5 -0.5
    //  b1 f0:  0.0 -1.0
    //
    //  b1 f1: -1.0  2.7
    //  b1 f1:  9.5  5.5
    //

    gpu_info_helper gpu_info;
    if (!gpu_info.get_engine_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    auto input = memory::allocate({  memory::format::yxfb_f16,{ 2,{ 2, 2 }, 2 }, 2 });
    auto output = memory::allocate({  memory::format::bfyx_f16,{ 2,{ 2, 2 }, 2 }, 2 });
    std::vector<float> subtract_val = { 0.5, 2.5 };

    set_values(input, {
        half_t(0x3C00), half_t(0x0000), // 1.f, 0.f,
        half_t(0x4500), half_t(0x3E00), // 5.f, 1.5f,

        half_t(0x4000), half_t(0x0000), // 2.f, 0.f,
        half_t(0x4600), half_t(0x4533), // 6.f, 5.2f,

        half_t(0x4200), half_t(0x3800), // 3.f, 0.5f,
        half_t(0x4700), half_t(0x4A00), // 7.f, 12.f,

        half_t(0x4400), half_t(0xB800), // 4.f, -0.5f,
        half_t(0x4800), half_t(0x4800)  // 8.f, 8.f
    });

    auto reorder = reorder::create({  output, input, subtract_val, true });

    execute({ reorder }).wait();

    half_t answers[16] = { half_t(0x3800), half_t(0x3E00), //  0.5f, 1.5f,
                           half_t(0x4100), half_t(0x4300), //  2.5f, 3.5f,
                            
                           half_t(0x4100), half_t(0x4300), //  2.5f, 3.5f,
                           half_t(0x4480), half_t(0x4580), //  4.5f, 5.5f,
                            
                           half_t(0xB800), half_t(0xB800), // -0.5f, -0.5f,
                           half_t(0x0000), half_t(0xBC00), //  0.0f, -1.0f,
                            
                           half_t(0xBC00), half_t(0x4166), // -1.0f,  2.7f,
                           half_t(0x48C0), half_t(0x4580)  //  9.5f,  5.5f
    };

    auto output_ptr = output.as<const memory&>().pointer<half_t>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(static_cast<uint16_t>(answers[i]), static_cast<uint16_t>(output_ptr[i])));
    }
}

TEST(reorder_gpu, basic_convert_f16_f32_f16) {
    //  Converts entire unambiguous range of FP16 numbers to FP32 and back.
    //
    //  Input               : 2x2x15873x1 (FP16)
    //  Intermediate        : 1x2x2x15873 (FP32) {different mem format but the same ordering because batch is 1}
    //  Output              : 2x2x15673x1 (FP16)
    //
    //  Output is expected to contain the same value as input in range of indices from 0x0000 to 0xF801.
    //

    gpu_info_helper gpu_info;
    if (!gpu_info.get_engine_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    std::vector<half_t> expected_values;
    expected_values.resize(0xF804);
    for (int i = 0; i < 0x7C00; ++i)
        expected_values[i] = half_t(i);          // norms/denorms/zero (positive).
    for (int i = 0x7C00; i < 0xF800; ++i)
        expected_values[i] = half_t(i + 0x0400); // norms/denorms (negative).
    expected_values[0x7C00] = half_t(0x0000);    // NOTE: do not do final test for negative 0 (-0).
    // Special values.
    expected_values[0xF800] = half_t(0x7C00);    // +infinity
    expected_values[0xF801] = half_t(0xFC00);    // -infinity
    // Special values (ambiguous ones).
    expected_values[0xF802] = half_t(0x8000);    // -0
    expected_values[0xF803] = half_t(0xFC12);    // A NaN (sample: -NaN.0x12).

    auto input = memory::allocate({memory::format::yxfb_f16,{1,{2, 2}, static_cast<uint32_t>(expected_values.size()) / 4}, 2});
    auto interm = memory::allocate({memory::format::byxf_f32,{1,{2, 2}, static_cast<uint32_t>(expected_values.size()) / 4}, 2});
    auto output = memory::allocate({memory::format::yxfb_f16,{1,{2, 2}, static_cast<uint32_t>(expected_values.size()) / 4}, 2});

    set_values(input, expected_values);

    auto reorder_f16_f32 = reorder::create({ input, interm });
    auto reorder_f32_f16 = reorder::create({ interm, output });

    execute({ reorder_f16_f32, reorder_f32_f16 }).wait();

    auto interm_ptr = interm.as<const memory&>().pointer<float>();
    // Sample positive.
    EXPECT_TRUE(are_equal(interm_ptr[0x3400], 0.25f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3800], 0.5f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3C00], 1.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4000], 2.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4400], 4.0f));
    // Sample negative.
    EXPECT_TRUE(are_equal(interm_ptr[0x3400 + 0x7C00], -0.25f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3800 + 0x7C00], -0.5f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3C00 + 0x7C00], -1.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4000 + 0x7C00], -2.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4400 + 0x7C00], -4.0f));
    // Special values.
    EXPECT_TRUE(are_equal(interm_ptr[0xF800], std::numeric_limits<float>::infinity()));
    EXPECT_TRUE(are_equal(interm_ptr[0xF801], -std::numeric_limits<float>::infinity()));
    EXPECT_TRUE(are_equal(interm_ptr[0xF802], -0.0f));
    EXPECT_TRUE(std::isnan(interm_ptr[0xF803]));


    auto output_ptr = output.as<const memory&>().pointer<half_t>();
    for (int i = 0; i < 0xF802; ++i) // NOTE: do not test for possibly ambiguous values of floating point (-0, NaNs).
    {
        EXPECT_TRUE(are_equal(static_cast<uint16_t>(expected_values[i]), static_cast<uint16_t>(output_ptr[i])));
    }
}