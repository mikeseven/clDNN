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

// memory->memory softmax
void example_softmax_forward() {
    using namespace neural;

    const uint32_t output_x  = 7, output_b  = 3,  // size of whole output buffer
                   input_x   = 6, input_b   = 2,  // size of whole input buffer
                   out_off_x = 0, out_off_b = 1,
                   out_siz_x = 5, out_siz_b = 2;  // size of area to do softmax after offset
    const uint32_t z = 1;
    const int32_t  in_off_x  = 1, in_off_b  = 0;

    float in_buffer[input_x*input_b];
    float out_buffer[output_x*output_b];
    // input buffer should be initialized with valid data

    auto input  = memory::create({engine::reference, memory::format::xb_f32, {input_b , std::vector<uint32_t>{input_x }, z}});
    auto output = memory::create({engine::reference, memory::format::xb_f32, {output_b, std::vector<uint32_t>{output_x}, z}});

    auto sftmax = normalization::softmax::create( {engine::reference,
                                                   output,
                                                   {out_off_b, std::vector<uint32_t>{out_off_x}, 0u},
                                                   {out_siz_b, std::vector<uint32_t>{out_siz_x}, 1u},
                                                   input,
                                                   {in_off_b, std::vector<int32_t>{in_off_x}, 0}
                                                  });

    execute({input(in_buffer), output(out_buffer), sftmax});
}

void example_softmax_backward(){
    // todo softmax bw
}