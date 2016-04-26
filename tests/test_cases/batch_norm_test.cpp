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

using namespace neural;

TEST(batch_normalization, trivial_forward_same_value_spatial_true) {

    // Random input size
    uint32_t input_size[4];

    uint32_t max_input_size = 16384; //2^14
    uint32_t possible_input_sizes[] = { 1, 2, 4, 8, 16, 32, 64 };
    int length = sizeof(possible_input_sizes) / sizeof(int);
    uint32_t random_size, i = 0;

    static std::mt19937 rng(1);
    std::uniform_int_distribution<int> dist(0, length - 1);

    while (i < 4) {
        random_size = possible_input_sizes[dist(rng)];
        if(random_size <= max_input_size){
            input_size[i] = random_size;
            max_input_size = max_input_size / random_size;
            i++;
        }
    };

    // Create input buffers.
    auto input               = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2], input_size[3] }, true });
    auto scale               = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });
    auto bias                = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });

    // Create output buffers.
    auto output              = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2], input_size[3] }, true });
    auto current_inv_std_dev = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });
    auto moving_average      = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });
    auto moving_inv_std_dev  = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });
    auto current_average     = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });

    auto& input_memory = input.as<const memory_obselote&>();
    auto& output_memory = output.as<const memory_obselote&>();
    auto& current_average_memory = current_average.as<const memory_obselote&>();

    // Initialize input buffers.
    input.as<const memory_obselote&>().fill<float>(1);
    scale.as<const memory_obselote&>().fill<float>(1);
    bias.as<const memory_obselote&>().fill<float>(0);


    // Create primitive.
    auto bn = normalization::batch_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, true, 1.0, std::numeric_limits<float>::epsilon()});

    // Run few times.
    for(i = 0; i < 3; ++i)
        execute({bn});

    for (i = 0; i < input_size[2]; ++i){
        EXPECT_EQ(0.0f, static_cast<float*>(output_memory.pointer)[i]);
    }
}

TEST(batch_normalization, trivial_forward_one_value_spatial_true) {

    // Random input size
    uint32_t input_size[4];

    uint32_t max_input_size = 16384; //2^14
    uint32_t possible_input_sizes[] = { 1, 2, 4, 8, 16, 32, 64 };
    int length = sizeof(possible_input_sizes) / sizeof(int);
    int non_zero_value;
    uint32_t random_size, i = 0, j;

    static std::mt19937 rng(1);
    std::uniform_int_distribution<int> dist(0, length - 1);

    while (i < 4) {
        random_size = possible_input_sizes[dist(rng)];
        if(random_size <= max_input_size){
            input_size[i] = random_size;
            max_input_size = max_input_size / random_size;
            i++;
        }
    }

    // Input size count
    uint32_t total_input_size = 1;
    for (i = 0; i < 4; i++) {
        total_input_size *= input_size[i];
    }

    // Create input buffers.
    auto input               = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2], input_size[3] }, true });
    auto scale               = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });
    auto bias                = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });

    // Create output buffers.
    auto output              = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2], input_size[3] }, true });
    auto current_inv_std_dev = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });
    auto moving_average      = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });
    auto moving_inv_std_dev  = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });
    auto current_average     = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, {             1,             1, input_size[2],             1 }, true });

    auto& input_memory = input.as<const memory_obselote&>();
    auto& output_memory = output.as<const memory_obselote&>();
    auto& current_average_memory = current_average.as<const memory_obselote&>();

    // Initialize input buffers.
    input.as<const memory_obselote&>().fill<float>(0);
    scale.as<const memory_obselote&>().fill<float>(1);
    bias.as<const memory_obselote&>().fill<float>(0);

    // Put non zero value in random place in input
    std::uniform_int_distribution<uint32_t> dist_input(0, total_input_size - 1);
    auto random_input_non_zero = dist_input(rng);
    static_cast<float*>(input_memory.pointer)[random_input_non_zero] = 10.0f;

    // Create primitive.
    auto bn = normalization::batch_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, true, 1.0, 1.0});

    // Run few times.
    for(i = 0; i < 3; ++i)
        execute({bn});

    // Find non zero value in avarages
    if (input_size[2] == 1) {
        non_zero_value = 0;
    }
    else {
        i = 0;
        while (random_input_non_zero >= i++ * input_size[3]) {
            non_zero_value = (i - 1) % input_size[2];
        }
    }

    // Count expected output
    float * expected_output = new float[total_input_size];
    for (i = 0; i < total_input_size; ++i) {
        expected_output[i] = 0;
    }

    float current_inv_std_dev_buffer = 0;
    float inv_num_average_over = (float) (1.0 / (input_size[0] * input_size[1] * input_size[3]));
    
    for (i = 0; i < input_size[0] * input_size[1] * input_size[3] - 1; i++) {
        current_inv_std_dev_buffer += pow(current_average_memory.get_value<float>(non_zero_value), 2.0f) * inv_num_average_over;
    }
    current_inv_std_dev_buffer += pow(10.0f - current_average_memory.get_value<float>(non_zero_value), 2.0f) * inv_num_average_over;
    current_inv_std_dev_buffer = pow(current_inv_std_dev_buffer + 1.0f, -0.5f);

    int iterator;
    for (i = 0; i < input_size[0] * input_size[1]; i++) {
        iterator = i * (total_input_size / (input_size[0] * input_size[1]));
        for (j = 0; j < input_size[3]; j++) {
            expected_output[iterator + (non_zero_value * input_size[3]) + j] = - current_average_memory.get_value<float>(non_zero_value) * current_inv_std_dev_buffer;
        }
    }
    expected_output[random_input_non_zero] = (10.0f - current_average_memory.get_value<float>(non_zero_value)) * current_inv_std_dev_buffer;

    for (i = 0; i < total_input_size; ++i) {
        EXPECT_EQ(expected_output[i], static_cast<float*>(output_memory.pointer)[i]);
    }
}

TEST(batch_normalization, trivial_forward_same_value_spatial_false) {

    // Random input size
    uint32_t input_size[4];

    uint32_t max_input_size = 16384; //2^14
    uint32_t possible_input_sizes[] = { 1, 2, 4, 8, 16, 32, 64};
    int length = sizeof(possible_input_sizes) / sizeof(int);
    uint32_t random_size, i = 0;

    static std::mt19937 rng(1);
    std::uniform_int_distribution<int> dist(0, length - 1);

    while (i < 4) {
        random_size = possible_input_sizes[dist(rng)];
        if(random_size <= max_input_size){
            input_size[i] = random_size;
            max_input_size = max_input_size / random_size;
            i++;
        }
    }

    // Input size count
    uint32_t total_avarage_size = 1;
    for (i = 0; i < 3; i++) {
        total_avarage_size *= input_size[i];
    }

    // Create input buffers.
    auto input               = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2], input_size[3] }, true });
    auto scale               = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });
    auto bias                = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });

    // Create output buffers.
    auto output              = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2], input_size[3] }, true });
    auto moving_average      = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });
    auto moving_inv_std_dev  = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });
    auto current_average     = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });
    auto current_inv_std_dev = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });

    auto& input_memory = input.as<const memory_obselote&>();
    auto& output_memory = output.as<const memory_obselote&>();
    auto& current_average_memory = current_average.as<const memory_obselote&>();

    // Initialize input buffers.
    input.as<const memory_obselote&>().fill<float>(1);
    scale.as<const memory_obselote&>().fill<float>(1);
    bias.as<const memory_obselote&>().fill<float>(0);


    // Create primitive.
    auto bn = normalization::batch_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, false, 1.0, std::numeric_limits<float>::epsilon()});

    // Run few times.
    for(i = 0; i < 3; ++i)
        execute({bn});

    for (i = 0; i < total_avarage_size; ++i){
        EXPECT_EQ(0.0f, static_cast<float*>(output_memory.pointer)[i]);
    }
}

TEST(batch_normalization, trivial_forward_one_value_spatial_false) {

    // Random input size
    uint32_t input_size[4];

    uint32_t max_input_size = 16384; //2^14
    uint32_t possible_input_sizes[] = { 1, 2, 4, 8, 16, 32, 64 };
    int length = sizeof(possible_input_sizes) / sizeof(int);
    int non_zero_value;
    uint32_t random_size, i = 0, j;

    static std::mt19937 rng(1);
    std::uniform_int_distribution<int> dist(0, length - 1);

    while (i < 4) {
        random_size = possible_input_sizes[dist(rng)];
        if(random_size <= max_input_size){
            input_size[i] = random_size;
            max_input_size = max_input_size / random_size;
            i++;
        }
    }

    // Input size count
    uint32_t total_average_size = 1;
    for (i = 0; i < 3; i++) {
        total_average_size *= input_size[i];
    }
    uint32_t total_input_size = total_average_size * input_size[3];

    // Create input buffers.
    auto input               = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2], input_size[3] }, true });
    auto scale               = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });
    auto bias                = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });

    // Create output buffers.
    auto output              = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2], input_size[3] }, true });
    auto current_inv_std_dev = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });
    auto moving_average      = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });
    auto moving_inv_std_dev  = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });
    auto current_average     = memory_obselote::create({ engine::cpu, memory_obselote::format::yxfb_f32, { input_size[0], input_size[1], input_size[2],             1 }, true });

    auto& input_memory = input.as<const memory_obselote&>();
    auto& output_memory = output.as<const memory_obselote&>();
    auto& current_average_memory = current_average.as<const memory_obselote&>();

    // Initialize input buffers.
    input.as<const memory_obselote&>().fill<float>(0);
    scale.as<const memory_obselote&>().fill<float>(1);
    bias.as<const memory_obselote&>().fill<float>(0);

    // Put non zero value in random place in input
    std::uniform_int_distribution<uint32_t> dist_input(0, total_input_size - 1);
    auto random_input_non_zero = dist_input(rng);
    static_cast<float*>(input_memory.pointer)[random_input_non_zero] = 10.0f;

    // Create primitive.
    auto bn = normalization::batch_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, false, 1.0, 1.0});

    // Run few times.
    for(i = 0; i < 3; ++i)
        execute({bn});

    float mean = (float) 10 / input_size[3];

    // Find non zero value in avarages
    if (input_size[2] == 1) {
        non_zero_value = 0;
    }
    else {
        i = 0;
        while (random_input_non_zero >= i++ * input_size[3]) {
            non_zero_value = (i - 1) % total_average_size;
        }
    }

    // Count expected output
    float * expected_output = new float[total_input_size];
    for (i = 0; i < total_input_size; ++i) {
        expected_output[i] = 0;
    }

    float current_inv_std_dev_buffer = 0;
    float inv_num_average_over = (float) (1.0 / input_size[3]);

    for (i = 0; i < input_size[3] - 1; i++) {
        current_inv_std_dev_buffer += pow(current_average_memory.get_value<float>(non_zero_value), 2.0f) * inv_num_average_over;
    }
    current_inv_std_dev_buffer += pow(10.0f - current_average_memory.get_value<float>(non_zero_value), 2.0f) * inv_num_average_over;
    current_inv_std_dev_buffer = pow(current_inv_std_dev_buffer + 1.0f, -0.5f);

    for (j = 0; j < input_size[3]; j++) {
        expected_output[(non_zero_value * input_size[3]) + j] = - current_average_memory.get_value<float>(non_zero_value) * current_inv_std_dev_buffer;
    }

    expected_output[random_input_non_zero] = (10.0f - current_average_memory.get_value<float>(non_zero_value)) * current_inv_std_dev_buffer;

    for (i = 0; i < total_input_size; ++i) {
        EXPECT_EQ(expected_output[i], static_cast<float*>(output_memory.pointer)[i]);
    }
}