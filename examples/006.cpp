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

#if 0
#include "api/neural.h"
#include <iostream>
#include <algorithm>

// choose fastest convolution & use it
void example_006() {
    using namespace neural;

    char  *in_ptr = nullptr;
    char *out_ptr = nullptr;

    // map & load resources
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {224, 224, 3,  24}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {224, 224, 96, 24}});
    auto weight = file::create({engine::cpu, "weight.nnb"});
    auto bias   = file::create({engine::cpu, "bias.nnb"});

    // query convolutions: any engine producing any data format
    auto result = convolution::query({engine::any, memory::format::any, input, weight, bias, padding::zero});
    for(auto &entry : result) std::cout
            << entry["engine"].u64() << ":" << entry["engine"].s()
            << "time = " << entry["time"].f32()*1000.0f << "ms, "
            << "energy = " << entry["energy"].f32() << "j" << std::endl;

#if 1
    // comparator/metric: the fastest
    auto compare = [](is_a_query_entry a, is_a_query_entry b) {
        return a["time"].f32()<b["time"].f32();
    };
#else
    // comparator/metric: performance/watt
    auto compare = [](is_a_query_entry a, is_a_query_entry b) {
        return a["time"].f32()/a["energy"].f32()<b["time"].f32()/b["energy"].f32();
    };
#endif

    // choose best & create it
    auto &best = std::min_element(result.begin(), result.end(), compare);
    primitive conv  = convolution::create(best->arguments);

    // execution
    execute({input(in_ptr), output(out_ptr), conv});
}
#endif