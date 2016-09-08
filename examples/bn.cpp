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
#include <limits>

using namespace neural;

void spatial_bn_trivial_example_forward_training_float()
{
    // Create input buffers.
    auto input               = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
    auto scale               = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto bias                = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});

    // Create output buffers.
    auto output              = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
    auto current_average     = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto current_inv_std_dev = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto moving_average      = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto moving_inv_std_dev  = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});

    // Initialize input buffers.
    fill<float>(input.as<const memory&>(), 0);
    fill<float>(scale.as<const memory&>(), 0);
    fill<float>(bias.as<const memory&>(), 0);

    // Create primitive.
    auto bn = normalization::batch_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, true, 0.0, std::numeric_limits<float>::epsilon()});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn}).wait();
}



void spatial_bn_trivial_example_inference_float()
{
    // Create input buffers.
    auto input       = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
    auto scale       = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto bias        = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto average     = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto inv_std_dev = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});

    // Create output buffer.
    auto output      = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});

    // Initialize input buffers.
    fill<float>(input.as<const memory&>(), 0);
    fill<float>(scale.as<const memory&>(), 0);
    fill<float>(bias.as<const memory&>(), 0);
    fill<float>(average.as<const memory&>(), 0);
    fill<float>(inv_std_dev.as<const memory&>(), 0);

    // Create primitive.
    auto bn = normalization::batch_inference::create({engine::reference, {output}, {input, scale, bias, average, inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn}).wait();
}

