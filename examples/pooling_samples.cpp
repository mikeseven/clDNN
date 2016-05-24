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

// memory->memory max pooling
void example_pooling_forward() {
    using namespace neural;

    const uint32_t output_y      = 7, output_x      = 7, output_z      = 3, output_b      = 3,  // size of whole output buffer
                   input_y       = 6, input_x       = 6, input_z       = 2, input_b       = 2,  // size of whole input buffer
                   out_off_y     = 1, out_off_x     = 2, out_off_z     = 1, out_off_b     = 0,
                   out_siz_y     = 5, out_siz_x     = 5, out_siz_z     = 2, out_siz_b     = 2,  // size of area to do pooling after offset
                   stride_y      = 1, stride_x      = 1, stride_z      = 1, stride_b      = 1,
                   pooling_siz_y = 2, pooling_siz_x = 2, pooling_siz_z = 1, pooling_siz_b = 1;  // size of pooling window

    const int32_t in_off_y = 0, in_off_x = 0, in_off_z = 0, in_off_b = 0;

    float in_buffer[input_y*input_x*input_z*input_b];
    float out_buffer[output_y*output_x*output_z*output_b];
    // input buffer should be initialized with valid data

    auto input  = memory::describe({engine::reference, memory::format::yxfb_f32, { input_b ,{ input_y, input_x }, input_z }});
    auto output = memory::describe({engine::reference, memory::format::yxfb_f32, { output_b,{output_y, output_x}, output_z}});

    auto pool  = pooling::create( {engine::reference,
                                    pooling::mode::max,
                                    output,
                                    {out_off_b, {out_off_y, out_off_x}, out_off_z},
                                    {out_siz_b, {out_siz_y, out_siz_x}, out_siz_z},
                                    input,
                                    { in_off_b, {in_off_y, in_off_x}, in_off_z},
                                    { stride_b, {stride_y, stride_x}, stride_z},
                                    {pooling_siz_b, {pooling_siz_y, pooling_siz_x}, pooling_siz_z},
                                    padding::zero}
                                  );

    execute({input(in_buffer), output(out_buffer), pool}).wait();
}

void example_pooling_backward(){
    // todo pooling bw
}