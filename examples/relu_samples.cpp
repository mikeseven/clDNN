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

// memory->memory relu
void example_relu_forward() {
    using namespace neural;
    float *data_buffer = nullptr;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});

    auto act    = relu::create({engine::reference, output, input});

    execute({input(data_buffer), output(data_buffer), act});
}

void example_relu_backward() {
    using namespace neural;

    auto forward_input  = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto forward_output_grad = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});

    auto forward_input_grad = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});

    auto act = relu_backward::create({engine::reference, {forward_input_grad}, {forward_output_grad, forward_input}});

    execute({act});
}
