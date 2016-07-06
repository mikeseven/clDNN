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
#include "tests/gtest/gtest.h"
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
#if 0 // no support yet
TEST(relu_f32_fw_gpu, basic) {
    // FAIL now, because we don't support using the same buffer as input and output
    EXPECT_EQ(false, true);
    return;

    const uint32_t y = 8, x = 8, f = 3, b = 2;

    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ b,{ y, x }, f } });
    auto output = memory::describe({ engine::gpu, memory::format::yxfb_f32,{ b,{ y, x }, f } });
    fill<float>(input.as<const memory&>());

    auto act = relu::create({ engine::gpu, output, input });
    auto buf = static_cast<float*>(input.as<const memory&>().pointer);
    // write output to input buffer
    execute({ output(buf), act }).wait();

    // multiply all positive intigers by -1
    for (size_t i = 0; i < y*x*f*b; ++i)
        buf[i] = (buf[i] > 0) ? -buf[i] : buf[i];

    execute({ act }).wait();

    bool result = false;
    // every element should be 0.0f
    for (size_t i = 0; i < y*x*f*b; ++i)
        result = result || buf[i];

    EXPECT_EQ(false, result);
}
#endif // NOT YET
TEST(relu_f32_fw_gpu, intrinsics_avx2) {
    const uint32_t y = 8, x = 8, f = 3, b = 2;

    // Optimized data
    auto input = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ b,{ y, x }, f } });
    auto output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ b,{ y, x }, f } });
    auto& input_memory = input.as<const memory&>();
    auto& output_memory = output.as<const memory&>();

    // Reference data
    auto ref_output = memory::allocate({ engine::gpu, memory::format::yxfb_f32,{ b,{ y, x }, f } });
    auto& ref_output_memory = ref_output.as<const memory&>();

    // Initialize input data
    fill<float>(input_memory);

    // Relu primitives
    auto opt_relu = relu::create({ engine::cpu, output, input });
    auto ref_relu = relu::create({ engine::gpu, ref_output, input });

    execute({ output, opt_relu }).wait();
    execute({ ref_output, ref_relu }).wait();

    for (size_t output_element = 0; output_element < output_memory.count(); ++output_element)
        EXPECT_EQ(true, tests::are_equal(static_cast<float*>(ref_output_memory.pointer)[output_element], static_cast<float*>(output_memory.pointer)[output_element]));
}