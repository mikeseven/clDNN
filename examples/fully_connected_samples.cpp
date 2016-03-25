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

void example_fully_connected() {
    using namespace neural;

    const uint32_t output_x  = 7, output_b  = 1,  // size of whole output buffer
                   input_x   = 6, input_b   = 1,  // size of whole input buffer
                   weight_x  = 6, weight_y  = 7;  // size of whole weights buffer

    float in_buffer[input_x*input_b];
    float out_buffer[output_x*output_b];
    float weight_buffer[weight_x*weight_y];

    auto input   = memory::create({ engine::cpu, memory::format::xb_f32,{ input_x,  input_b  } });
    auto output  = memory::create({ engine::cpu, memory::format::xb_f32,{ output_x, output_b } });
    auto weights = memory::create({ engine::cpu, memory::format::xb_f32,{ weight_x, weight_y } });

    auto act = fully_connected::create({ engine::reference,
                                         output,
                                         input,
                                         weights }
                                      );

    execute({ input(in_buffer), output(out_buffer), weights(weight_buffer), act });
}