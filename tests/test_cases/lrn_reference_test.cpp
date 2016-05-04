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
#include "multidimensional_counter.h"
#include "test_utils/test_utils.h"
#include <iostream>

TEST(local_response_normalization, lrn_test) {

    using namespace neural;
    using namespace tests;

    // test initialization

    // input-output parameters:

    const uint32_t px = 2, py = 2, pb = 1, pf = 7, psize = 3;

    std::initializer_list<float> input_oracle_init = {
        -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f,   // b=0, x=0, y=0
        -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f,   // b=0, x=1, y=0
        -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f,   // b=0, x=0, y=1
        -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};  // b=0, x=1, y=1

    std::initializer_list<float> output_oracle_init = {
        -0.99998f, -0.49999f, 0.00000f, 0.49999f, 0.99995f, 1.49984f, 1.99981f,   // b=0, x=0, y=0
        -0.99998f, -0.49999f, 0.00000f, 0.49999f, 0.99995f, 1.49984f, 1.99981f,   // b=0, x=1, y=0
        -0.99998f, -0.49999f, 0.00000f, 0.49999f, 0.99995f, 1.49984f, 1.99981f,   // b=0, x=0, y=1
        -0.99998f, -0.49999f, 0.00000f, 0.49999f, 0.99995f, 1.49984f, 1.99981f};  // b=0, x=1, y=1}

    // lrn parameters:
    const float pk = 1.f, palpha = 0.00002f, pbeta = 0.75f;

    auto input = memory::create({ engine::reference, memory::format::yxfb_f32,{ pb,{ px, py }, pf }, true });
    auto output = memory::create({ engine::reference, memory::format::yxfb_f32,{ pb,{ px, py }, pf }, true });
    auto output_oracle = memory::create({ engine::reference, memory::format::yxfb_f32,{ pb,{ px, py }, pf }, true });

    set_values(input, input_oracle_init);
    set_values(output_oracle, output_oracle_init);

    auto lrn = normalization::response::create({ engine::reference, output, input, psize, padding::zero, pk, palpha, pbeta });

    // ------------------------------------------------------------------------------------------------
    // test run
    execute({ lrn });
    
    // analysis of results
    float* buff = nullptr;
    float* buff_oracle = nullptr;

    bool   result = true;

    try {

        buff = static_cast<float*>(output.as<const memory&>().pointer);
        buff_oracle = static_cast<float*>(output_oracle.as<const memory&>().pointer);

        for (size_t i = 0; i < px*py*pb*pf; ++i) {
            result &= values_comparison(buff[i], buff_oracle[i], 5.2e-04F);
        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }

    EXPECT_EQ(true, result);
    // ------------------------------------------------------------------------------------------------
    // test clean

}
