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
#include "memory_utils.h"
#include <iostream>

TEST(local_response_normalization, lrn_cpu_avx2_test) {

    using namespace neural;
    using namespace tests;

    // ------------------------------------------------------------------------------------------------
    // TEST INITIALIZATION
    // Memory descriptors for reference lrn


    // input-output parameters:
    const uint32_t param_x = 2, param_y = 2, param_b = 24, param_f = 16, param_size = 5;
    // lrn parameters:
    const float param_k = 1.0f, param_alpha = 1.0f, param_beta = 0.75f;

    // reference
    auto input_reference = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ param_b,{ param_x, param_y }, param_f } });
    auto output_reference = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ param_b,{ param_x, param_y }, param_f } });
    fill<float>(input_reference);

    auto lrn_reference = normalization::response::create({engine::reference, output_reference, input_reference, param_size, padding::zero, param_k, param_alpha, param_beta});

    // optimized
    auto input_optimized = memory::allocate({ engine::reference, memory::format::byxf_f32,{ param_b,{ param_x, param_y }, param_f } });
    auto output_optimized = memory::allocate({ engine::reference, memory::format::byxf_f32,{ param_b,{ param_x, param_y }, param_f } });
    auto output_optimized_in_reference_format = memory::allocate({ engine::reference, memory::format::yxfb_f32,{ param_b,{ param_x, param_y }, param_f } });

    auto reorder_input_reference_to_optimized = reorder::create({ engine::reference, input_reference, input_optimized });
    auto reorder_output_optimized_to_reference_format = reorder::create({ engine::reference, output_optimized, output_optimized_in_reference_format });

    auto lrn_optimized = normalization::response::create({ engine::cpu, output_optimized, input_optimized, param_size, padding::zero, param_k, param_alpha, param_beta });
    
    // ------------------------------------------------------------------------------------------------
    // TEST RUN
    execute({ lrn_reference, reorder_input_reference_to_optimized, lrn_optimized, reorder_output_optimized_to_reference_format }).wait();

    // analysis of results
    bool   result = true;

    try {

        auto buff = static_cast<float*>(output_optimized_in_reference_format.as<const memory&>().pointer);
        auto buff_reference = static_cast<float*>(output_reference.as<const memory&>().pointer);

        for (size_t i = 0; i < param_x*param_y*param_b*param_f; ++i) {
            EXPECT_EQ(true, tests::are_equal(buff_reference[i], buff[i], 1e-04F, 1e-04F, 1e-04F)) << "at index " << i;
        }
    }
    catch (const std::exception& E) {
        std::cout << E.what() << std::endl;
    }

    EXPECT_EQ(true, result);
    // ------------------------------------------------------------------------------------------------
    // TEST CLEAN

}
