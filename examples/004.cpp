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

// AlexNet with weights & biases from file
void example_004() {
    char  *input_buffer = nullptr;
    char *output_buffer = nullptr;

    uint32_t batch_size = 24;

    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {227, 227, 3,  batch_size}});
    auto output = memory::create({engine::cpu, memory::format::xb_f32, {1000, batch_size}});

    // [227x227x3xB] convolution->relu->pooling->lrn [27x27x96xB]
    auto conv1  = convolution::create({engine::cpu, memory::format::yxfb_f32, input, 2, file::create({engine::cpu, "weight1.nnb"}), file::create({engine::cpu, "bias1.nnb"}), padding::zero});
    auto relu1  = relu::create({engine::cpu, memory::format::yxfb_f32, conv1});
    auto pool1  = pooling::create({engine::cpu, pooling::mode::max, memory::format::yxfb_f32, relu1, 3, 2, padding::zero});
    auto lrn1   = normalization::response::create({engine::cpu, memory::format::yxfb_f32, pool1, 5, padding::zero, 1.0f, 0.00002f, 0.75f });

    // [27x27x96xB] convolution->relu->pooling->lrn [13x13x256xB]
    auto conv2  = convolution::create({engine::cpu, memory::format::yxfb_f32, lrn1, file::create({engine::cpu, "weight2.nnb"}), file::create({engine::cpu, "bias2.nnb"}), padding::zero});
    auto relu2  = relu::create({engine::cpu, memory::format::yxfb_f32, conv2});
    auto pool2  = pooling::create({engine::cpu, pooling::mode::max, memory::format::yxfb_f32, relu2, 3, 2, padding::zero});
    auto lrn2   = normalization::response::create({engine::cpu, memory::format::yxfb_f32, pool2, 5, padding::zero, 1.0f, 0.00002f, 0.75f });

    // [13x13x256xB] convolution->relu [13x13x384xB]
    auto conv3  = convolution::create({engine::cpu, memory::format::yxfb_f32, lrn2, file::create({engine::cpu, "weight3.nnb"}), file::create({engine::cpu, "bias3.nnb"}), padding::zero});
    auto relu3  = relu::create({engine::cpu, memory::format::yxfb_f32, conv3});

    // [13x13x384xB] convolution->relu [13x13x256xB]
    auto conv4  = convolution::create({engine::cpu, memory::format::yxfb_f32, relu3, file::create({engine::cpu, "weight4.nnb"}), file::create({engine::cpu, "bias4.nnb"}), padding::zero});
    auto relu4  = relu::create({engine::cpu, memory::format::yxfb_f32, conv4});

    // [13x13x256xB] convolution->relu->pooling [6x6x256xB]
    auto conv5  = convolution::create({engine::cpu, memory::format::yxfb_f32, relu4, file::create({engine::cpu, "weight5.nnb"}), file::create({engine::cpu, "bias5.nnb"}), padding::zero});
    auto relu5  = relu::create({engine::cpu, memory::format::yxfb_f32, conv5});
    auto pool5  = pooling::create({engine::cpu, pooling::mode::max, memory::format::yxfb_f32, relu5, 3, 2, padding::zero});

    // [6x6x256xB] fully_connected->relu [4096xB]
    auto fc6    = fully_connected::create({engine::cpu, memory::format::yxfb_f32, pool5, file::create({engine::cpu, "weight6.nnb"}), file::create({engine::cpu, "bias6.nnb"})});
    auto relu6  = relu::create({engine::cpu, memory::format::yxfb_f32, fc6});

    // [4096xB] convolution->relu [4096xB]
    auto fc7    = fully_connected::create({engine::cpu, memory::format::yxfb_f32, relu6, file::create({engine::cpu, "weight7.nnb"}), file::create({engine::cpu, "bias7.nnb"})});
    auto relu7  = relu::create({engine::cpu, memory::format::yxfb_f32, fc7});

    // [4096xB] convolution->relu [1000xB]
    auto fc8    = fully_connected::create({engine::cpu, memory::format::yxfb_f32, relu7, file::create({engine::cpu, "weight8.nnb"}), file::create({engine::cpu, "bias8.nnb"})});
    auto relu8  = relu::create({engine::cpu, memory::format::yxfb_f32, fc8});

    // [1000xB] softmax [1000xB]
    auto sftmax = normalization::softmax::create({engine::cpu, output, relu8});

    execute({input(input_buffer), output(input_buffer), conv1, relu1, pool1, lrn1, conv2, relu2, pool2, lrn2, conv3, relu3, conv4, relu4, conv5, relu5, pool5, fc6, relu6, fc7, relu7, fc8, relu8, sftmax}).sync();
}
#endif