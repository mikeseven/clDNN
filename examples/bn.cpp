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
        execute({bn}).sync();
}

void spatial_bn_trivial_example_backward_training_float()
{
    // Create input buffers.
    auto forward_input       = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
    auto forward_scale       = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto forward_bias        = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto output_grad         = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
    auto current_mean        = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto current_inv_std_dev = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});

    // Create output buffers.
    auto input_grad = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
    auto scale_grad = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto bias_grad  = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});

    // Initialize input buffers.
    fill<float>(forward_input.as<const memory&>(), 0);
    fill<float>(forward_scale.as<const memory&>(), 0);
    fill<float>(forward_bias.as<const memory&>(), 0);
    fill<float>(output_grad.as<const memory&>(), 0);
    fill<float>(current_mean.as<const memory&>(), 0);
    fill<float>(current_inv_std_dev.as<const memory&>(), 0);

    // Create primitive.
    auto bn = normalization::batch_training_backward::create({engine::reference, {input_grad, scale_grad, bias_grad}, {forward_input, forward_scale, forward_bias, output_grad, current_mean, current_inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn}).sync();
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
        execute({bn}).sync();
}

void spatial_bn_complex_example_training_float()
{
    // TODO: after string keys have been implemented, move buffer creation to primitives' constructors and reference them by primitive-string lookups.

    // Create data buffers that have to be initialized before training starts.
    auto forward_input       = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
    auto forward_scale       = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto forward_bias        = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});

    // Create intermediate buffers that will be computed during forward training pass.
    auto current_mean        = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto current_inv_std_dev = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto moving_mean         = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto moving_inv_std_dev  = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});

    // Create output buffers.
    auto forward_output      = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
    //auto forward_output_grad = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}}); // same as forward_output
    auto forward_input_grad  = memory::allocate({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
    auto forward_scale_grad  = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
    auto forward_bias_grad   = memory::allocate({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});

    // Initialize input buffers for forward training primitive which will initialize other buffers.
    fill<float>(forward_input.as<const memory&>(), 0);
    fill<float>(forward_scale.as<const memory&>(), 0);
    fill<float>(forward_bias.as<const memory&>(), 0);

    // Create primitives.
    auto bn_train_fw  = normalization::batch_training_forward::create({engine::reference, {forward_output, current_mean, current_inv_std_dev, moving_mean, moving_inv_std_dev}, {forward_input, forward_scale, forward_bias}, true, 0.0, std::numeric_limits<float>::epsilon()});
    auto bn_train_bck = normalization::batch_training_backward::create({engine::reference, {forward_input_grad, forward_scale_grad, forward_bias_grad}, {forward_input, forward_scale, forward_bias, forward_output, current_mean, current_inv_std_dev}, true});
    auto bn_infer     = normalization::batch_inference::create({engine::reference, {forward_output}, {forward_input, forward_scale, forward_bias, moving_mean, moving_inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
    {
        execute({bn_train_fw}).sync();

        // TODO: add some simple error function instead of directly connecting forward and backward pass by forward_output primitive
        // ....

        execute({bn_train_bck}).sync();

        // TODO: add SGD primitive
        // ...

        // TODO: after above is completed, whole training stack should be done via single execute() call.
    }

    // Run few times.
    for(int i = 0; i < 3; ++i)
    {
        execute({bn_infer}).sync();
    }
}
