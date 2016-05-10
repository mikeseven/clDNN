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
//  Filter : 2x1x2x1
//  Input  : 1x1x2x1
//  Output : 1x2x1x1
//
//  Input:
//  1.0    2.0
//
// Filter:
//   1.0    2.0  f=0
//  -1.0   -2.0  f=1
//
//  Bias:
//  0.1 -0.2
//
//  Output:
//   5.1  f=0
//  -5.2  f=1

    auto input   = memory::create({ engine::reference, memory::format::yxfb_f32,{1 ,{2, 1}, 1}, true });
    auto output  = memory::create({ engine::reference, memory::format::yxfb_f32,{1 ,{1, 1}, 2}, true });
    auto weights = memory::create({ engine::reference, memory::format::oiyx_f32,{1 ,{2, 1},{2, 1}}, true });
    auto biases  = memory::create({ engine::reference, memory::format::   x_f32,{1 ,{{2}}, 1}, true });

    set_values(input,   { 1.0f, 2.0f });
    set_values(weights, { 1.0f, 2.0f, -1.0f, -2.0f });
    set_values(biases,  { 0.1f, -0.2f});

    auto conv = convolution::create({ engine::reference, output, input, { 1, {5, 5}, 1 }, weights, biases, padding::zero });

    execute({ conv });

    auto& output_memory = output.as<const memory&>();
    std::cout << (5.1f == get_value<float>(output_memory, 0));
    std::cout << (-5.2f== get_value<float>(output_memory, 1));
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
