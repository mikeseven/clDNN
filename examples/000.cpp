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

// memory->memory convolution with weights & biases from file
void example_000() {
    char *in_ptr = nullptr, *out_ptr = nullptr;
    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {3,  {224, 224}, 24}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {96, {224, 224}, 24}});
    auto weight = file::create({engine::cpu, "weight.nnb"});
    auto bias   = file::create({engine::cpu, "bias.nnb"});
    auto conv  = convolution::create({engine::cpu, output, input, weight, bias, padding::zero});

    conv["engine"].s();     // std::string("cpu")
    conv["engine"].u64();   // uint64_t(1)
    conv["time"].f32();     // float(0.000001f)
    conv["inputs"].u32();   // uint32_t(3)
    conv["input0"].s();     // std::string("input")
    conv["input1"].s();     // std::string("weight")
    conv["input2"].s();     // std::string("bias")
    conv["name"].s();       // std::string("convolution")
    conv["info_short"].s(); // std::string("direct convolution")

    execute({input(in_ptr), output(out_ptr), conv});
}