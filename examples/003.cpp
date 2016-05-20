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

#if 0
// convolution->relu->pooling->lrn with weights & biases from file
void example_003() {
    char *data_buffer = nullptr;
    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {3,  {224, 224}, 24}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {96, {112, 112}, 24}});
    auto weight = file::create({engine::cpu, "weight.nnb"});
    auto bias   = file::create({engine::cpu, "bias.nnb"});

    auto conv   = convolution::create({engine::cpu, memory::format::yxfb_f32, input, weight, bias, padding::zero});
    auto act    = relu::create({engine::cpu, memory::format::yxfb_f32, conv});
    auto pool   = pooling::create({engine::cpu, pooling::mode::max, memory::format::yxfb_f32, act, 3, 2, padding::zero});
    auto lrn    = normalization::response::create({engine::cpu, output, pool, 5, padding::zero, 1.0f, 0.00002f, 0.75f });

    execute({input(data_buffer), output(data_buffer), conv, act, pool, lrn}).sync();
}

#endif