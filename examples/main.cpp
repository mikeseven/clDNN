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
#include <iostream>





#include "tests/test_utils/test_utils.h"
#include "memory_utils.h"
void TEST() {
    using namespace neural;
    using namespace tests;
//  Filter : 1x3x2x2x1
//  Input  : 1x2x2x1
//  Output : 1x3x1x1
//
//  Input:
//  1.0    2.0  f=0
//  3.0    4.0  f=1
//
// Filter:
//   1.0    2.0  ifm=0  ofm=0
//   3.0    4.0  ifm=1
//
//   5.0    6.0  ifm=0  ofm=1
//   7.0    8.0  ifm=1
//
//   9.0   10.0  ifm=0  ofm=2
//  11.0   12.0  ifm=1
//  Bias:
//   -5     -6     -7
//
//  Output:
//   25.0  f=0
//   70.0  f=1
//  110.0  f=2

    auto input   = memory::create({ engine::reference, memory::format::yxfb_f32,{1 ,{2, 1}, 2}, true });
    auto output  = memory::create({ engine::reference, memory::format::yxfb_f32,{1 ,{1, 1}, 3}, true });
    auto weights = memory::create({ engine::reference, memory::format::oiyx_f32,{1 ,{2, 1},{3, 2}}, true });
    auto biases  = memory::create({ engine::reference, memory::format::   x_f32,{1 ,{{3}}, 1}, true });

    set_values(input,   { 1.0f, 3.0f, 2.0f, 4.0f });
    set_values(weights, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });
    set_values(biases,  { -5.0f, -6.0f, -7.0f});

    auto conv = convolution::create({ engine::reference, output, input, { 1, {5, 5}, 1 }, weights, biases, padding::zero });

    execute({ conv });

    auto& output_memory = output.as<const memory&>();
    std::cout << (25.0f  == get_value<float>(output_memory, 0));
    std::cout << (64.0f == get_value<float>(output_memory, 1));
    std::cout << (113.0f == get_value<float>(output_memory, 2));
}

int main()
{
    extern void example_lrn_forward();
    try {
        TEST();
        //example_lrn_forward();
    }
    catch (std::exception &e) {
        std::cerr << e.what();
    }
    catch (...) {
        std::cerr << "Unknown exceptions.";
    }
    return 0;
}
