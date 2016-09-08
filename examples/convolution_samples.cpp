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
#include "memory_utils.h"

// memory->memory convolution
void example_convolution_ref_forward() {
    using namespace neural;

    const uint32_t output_y    = 16,
                   output_x    = 16,
                   output_z    = 32,
                   output_b    = 24,    // size of whole output buffer

                   input_y     = 32,
                   input_x     = 32,
                   input_z     = 8,
                   input_b     = 24,    // size of whole input buffer

                   stride_y    = 2,
                   stride_x    = 2,
                   stride_z    = 1,
                   stride_b    = 1,

                   conv_size_y = 2,
                   conv_size_x = 2,
                   conv_size_ifm = 4,
                   conv_size_ofm = 1;    // size of convolution window

//    const int32_t in_off_y = 0, in_off_x = 0, in_off_z = 0, in_off_b = 0;

    auto eng    = engine::reference;
    auto input  = memory::allocate({eng, memory::format::yxfb_f32, { input_b  , {input_y    , input_x    }, input_z                       }});
    auto output = memory::allocate({eng, memory::format::yxfb_f32, { output_b , {output_y   , output_x   }, output_z                      }});
    auto weights= memory::allocate({eng, memory::format::oiyx_f32, { 1        , {conv_size_y, conv_size_x}, {conv_size_ofm, conv_size_ifm}}});
    auto biases = memory::allocate({eng, memory::format::   x_f32, { 1        , {{output_z}}              , 1                             }});

    // buffers should be initialized with valid data
    fill(input.as  <const memory&>(), 1.0f);
    fill(output.as <const memory&>(), 1.0f);
    fill(weights.as<const memory&>(), 1.0f);
    fill(biases.as <const memory&>(), 1.0f);

    auto conv   = convolution::create( {eng,
                                        output,
//                                        {out_off_b, {out_off_y, out_off_x}, out_off_z},
//                                        {out_siz_b, {out_siz_y, out_siz_x}, out_siz_z},
                                        {input, weights, biases},
//                                        {in_off_b, {in_off_y, in_off_x}, in_off_z},
                                        {stride_b, {stride_y, stride_x}, stride_z},
                                        padding::zero}
                                      );

    execute({conv}).wait();
}

void example_convolution_cpu_forward() {
    using namespace neural;

    const uint32_t output_y    = 6,
                   output_x    = 6,
                   output_z    = 16,
                   output_b    = 1,    // size of whole output buffer

                   input_y     = 8+5,
                   input_x     = 8+5,
                   input_z     = 1,
                   input_b     = 1,    // size of whole input buffer

                   stride_y    = 1,
                   stride_x    = 1,
                   stride_z    = 1,
                   stride_b    = 1,

                   conv_size_y = 8,
                   conv_size_x = 8,
                   conv_size_ifm = input_z,
                   conv_size_ofm = output_z;    // size of convolution window

    auto eng    = engine::cpu;
    auto input  = memory::allocate({eng, memory::format::yxfb_f32, { input_b , {input_y , input_x }, input_z }});
    auto output = memory::allocate({eng, memory::format::yxfb_f32, { output_b, {output_y, output_x}, output_z}});
    auto weights= memory::allocate({eng, memory::format::oiyx_f32, { 1       , {conv_size_y, conv_size_x}, {conv_size_ofm, conv_size_ifm}}});
    auto biases = memory::allocate({eng, memory::format::   x_f32, { 1       , {{output_z}}        , 1       }});

    // buffers should be initialized with valid data
    fill(input.as<const memory&>()  , 1.0f);
    fill(output.as<const memory&>() , 1.0f);
    fill(weights.as<const memory&>(), 1.0f);
    fill(biases.as<const memory&>() , 1.0f);

    auto conv   = convolution::create( {eng,
                                        output,
                                        {input, weights, biases},
                                        {stride_b, {stride_y, stride_x}, stride_z},
                                        padding::zero}
                                      );

    execute({conv}).wait();
}

