#include "neural.h"

using namespace neural;

void spatial_bn_trivial_example_forward_training_float() 
{
    // Create data buffers.
    auto input               = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto scale               = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto bias                = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    auto output              = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto current_average     = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto moving_average      = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto moving_inv_std_dev  = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    // Initialize input buffers.
    memset(input.as<const memory&>().pointer, 0, input.as<const memory&>().count() * memory::traits(input.as<const memory&>().argument.format).type->size);
    memset(scale.as<const memory&>().pointer, 0, scale.as<const memory&>().count() * memory::traits(scale.as<const memory&>().argument.format).type->size);
    memset( bias.as<const memory&>().pointer, 0,  bias.as<const memory&>().count() * memory::traits( bias.as<const memory&>().argument.format).type->size);

    // Create primitive.
    auto bn = batch_normalization_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, true, 0.0, FLT_EPSILON});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_forward_training_double() 
{
    // Create data buffers.
    auto input               = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto scale               = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto bias                = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    auto output              = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto current_average     = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto moving_average      = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto moving_inv_std_dev  = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Initialize input buffers.
    memset(input.as<const memory&>().pointer, 0, input.as<const memory&>().count() * memory::traits(input.as<const memory&>().argument.format).type->size);
    memset(scale.as<const memory&>().pointer, 0, scale.as<const memory&>().count() * memory::traits(scale.as<const memory&>().argument.format).type->size);
    memset( bias.as<const memory&>().pointer, 0,  bias.as<const memory&>().count() * memory::traits( bias.as<const memory&>().argument.format).type->size);

    // Create primitive.
    auto bn = batch_normalization_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, true, 0.0, DBL_EPSILON});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_backward_training_float() 
{
    // Create data buffers.
    auto forward_input       = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto forward_scale       = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto forward_bias        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto output_grad         = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto current_mean        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    auto input_grad = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto scale_grad = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto bias_grad  = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    // Initialize input buffers.
    memset(      forward_input.as<const memory&>().pointer, 0,       forward_input.as<const memory&>().count() * memory::traits(      forward_input.as<const memory&>().argument.format).type->size);
    memset(      forward_scale.as<const memory&>().pointer, 0,       forward_scale.as<const memory&>().count() * memory::traits(      forward_scale.as<const memory&>().argument.format).type->size);
    memset(       forward_bias.as<const memory&>().pointer, 0,        forward_bias.as<const memory&>().count() * memory::traits(       forward_bias.as<const memory&>().argument.format).type->size);
    memset(        output_grad.as<const memory&>().pointer, 0,         output_grad.as<const memory&>().count() * memory::traits(        output_grad.as<const memory&>().argument.format).type->size);
    memset(       current_mean.as<const memory&>().pointer, 0,        current_mean.as<const memory&>().count() * memory::traits(       current_mean.as<const memory&>().argument.format).type->size);
    memset(current_inv_std_dev.as<const memory&>().pointer, 0, current_inv_std_dev.as<const memory&>().count() * memory::traits(current_inv_std_dev.as<const memory&>().argument.format).type->size);

    // Create primitive.
    auto bn = batch_normalization_training_backward::create({engine::reference, {input_grad, scale_grad, bias_grad}, {forward_input, forward_scale, forward_bias, output_grad, current_mean, current_inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_backward_training_double() 
{
    // Create data buffers.
    auto forward_input       = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto forward_scale       = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto forward_bias        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto output_grad         = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto current_mean        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    auto input_grad = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto scale_grad = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto bias_grad  = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Initialize input buffers.
    memset(      forward_input.as<const memory&>().pointer, 0,       forward_input.as<const memory&>().count() * memory::traits(      forward_input.as<const memory&>().argument.format).type->size);
    memset(      forward_scale.as<const memory&>().pointer, 0,       forward_scale.as<const memory&>().count() * memory::traits(      forward_scale.as<const memory&>().argument.format).type->size);
    memset(       forward_bias.as<const memory&>().pointer, 0,        forward_bias.as<const memory&>().count() * memory::traits(       forward_bias.as<const memory&>().argument.format).type->size);
    memset(        output_grad.as<const memory&>().pointer, 0,         output_grad.as<const memory&>().count() * memory::traits(        output_grad.as<const memory&>().argument.format).type->size);
    memset(       current_mean.as<const memory&>().pointer, 0,        current_mean.as<const memory&>().count() * memory::traits(       current_mean.as<const memory&>().argument.format).type->size);
    memset(current_inv_std_dev.as<const memory&>().pointer, 0, current_inv_std_dev.as<const memory&>().count() * memory::traits(current_inv_std_dev.as<const memory&>().argument.format).type->size);

    // Create primitive.
    auto bn = batch_normalization_training_backward::create({engine::reference, {input_grad, scale_grad, bias_grad}, {forward_input, forward_scale, forward_bias, output_grad, current_mean, current_inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_inference_float() 
{
    // Create data buffers.
    auto input       = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto scale       = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto bias        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto average     = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    auto output      = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});

    // Initialize input buffers.
    memset(      input.as<const memory&>().pointer, 0,       input.as<const memory&>().count() * memory::traits(      input.as<const memory&>().argument.format).type->size);
    memset(      scale.as<const memory&>().pointer, 0,       scale.as<const memory&>().count() * memory::traits(      scale.as<const memory&>().argument.format).type->size);
    memset(       bias.as<const memory&>().pointer, 0,        bias.as<const memory&>().count() * memory::traits(       bias.as<const memory&>().argument.format).type->size);
    memset(    average.as<const memory&>().pointer, 0,     average.as<const memory&>().count() * memory::traits(    average.as<const memory&>().argument.format).type->size);
    memset(inv_std_dev.as<const memory&>().pointer, 0, inv_std_dev.as<const memory&>().count() * memory::traits(inv_std_dev.as<const memory&>().argument.format).type->size);

    // Create primitive.
    auto bn = batch_normalization_inference::create({engine::reference, {output}, {input, scale, bias, average, inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_inference_double() 
{
    // Create data buffers.
    auto input       = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto scale       = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto bias        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto average     = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    auto output      = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});

    // Initialize input buffers.
    memset(      input.as<const memory&>().pointer, 0,       input.as<const memory&>().count() * memory::traits(      input.as<const memory&>().argument.format).type->size);
    memset(      scale.as<const memory&>().pointer, 0,       scale.as<const memory&>().count() * memory::traits(      scale.as<const memory&>().argument.format).type->size);
    memset(       bias.as<const memory&>().pointer, 0,        bias.as<const memory&>().count() * memory::traits(       bias.as<const memory&>().argument.format).type->size);
    memset(    average.as<const memory&>().pointer, 0,     average.as<const memory&>().count() * memory::traits(    average.as<const memory&>().argument.format).type->size);
    memset(inv_std_dev.as<const memory&>().pointer, 0, inv_std_dev.as<const memory&>().count() * memory::traits(inv_std_dev.as<const memory&>().argument.format).type->size);

    // Create primitive.
    auto bn = batch_normalization_inference::create({engine::reference, {output}, {input, scale, bias, average, inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}