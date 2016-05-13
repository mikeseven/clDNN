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
void example_convolution_exec_resource_forward() {
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

    auto engine_resource = execution_resource_cpu::create({std::thread::hardware_concurrency()});

    auto input  = memory::create({engine::cpu, memory::format::byxf_f32, { input_b      , {input_y    , input_x    }, input_z      }, true});
    auto output = memory::create({engine::cpu, memory::format::byxf_f32, { output_b     , {output_y   , output_x   }, output_z     }, true});
    auto weights= memory::create({engine::cpu, memory::format::byxf_f32, { conv_size_ofm, {conv_size_y, conv_size_x}, conv_size_ifm}, true});
    auto biases = memory::create({engine::cpu, memory::format::   x_f32, { 1            , {{output_z}}              , 1            }, true});

    // buffers should be initialized with valid data
    fill(input.as  <const memory&>(), 1.0f);
    fill(output.as <const memory&>(), 1.0f);
    fill(weights.as<const memory&>(), 1.0f);
    fill(biases.as <const memory&>(), 1.0f);

    auto conv   = convolution::create( {engine::cpu,
                                        output,
                                        input,
                                        {stride_b, {stride_y, stride_x}, stride_z},
                                        weights,
                                        biases,
                                        padding::zero}
                                      );

    execute({conv}, engine_resource);
}