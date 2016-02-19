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
    auto data_size = memory::traits(input.as<const memory&>().argument.format).type->size;
    memset(input.as<const memory&>().pointer, 0, input.as<const memory&>().count() * data_size);
    memset(scale.as<const memory&>().pointer, 0, scale.as<const memory&>().count() * data_size);
    memset( bias.as<const memory&>().pointer, 0,  bias.as<const memory&>().count() * data_size);

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
    auto data_size = memory::traits(input.as<const memory&>().argument.format).type->size;
    memset(input.as<const memory&>().pointer, 0, input.as<const memory&>().count() * data_size);
    memset(scale.as<const memory&>().pointer, 0, scale.as<const memory&>().count() * data_size);
    memset( bias.as<const memory&>().pointer, 0,  bias.as<const memory&>().count() * data_size);

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
    auto data_size = memory::traits(forward_input.as<const memory&>().argument.format).type->size;
    memset(      forward_input.as<const memory&>().pointer, 0,       forward_input.as<const memory&>().count() * data_size);
    memset(      forward_scale.as<const memory&>().pointer, 0,       forward_scale.as<const memory&>().count() * data_size);
    memset(       forward_bias.as<const memory&>().pointer, 0,        forward_bias.as<const memory&>().count() * data_size);
    memset(        output_grad.as<const memory&>().pointer, 0,         output_grad.as<const memory&>().count() * data_size);
    memset(       current_mean.as<const memory&>().pointer, 0,        current_mean.as<const memory&>().count() * data_size);
    memset(current_inv_std_dev.as<const memory&>().pointer, 0, current_inv_std_dev.as<const memory&>().count() * data_size);

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
    auto data_size = memory::traits(forward_input.as<const memory&>().argument.format).type->size;
    memset(      forward_input.as<const memory&>().pointer, 0,       forward_input.as<const memory&>().count() * data_size);
    memset(      forward_scale.as<const memory&>().pointer, 0,       forward_scale.as<const memory&>().count() * data_size);
    memset(       forward_bias.as<const memory&>().pointer, 0,        forward_bias.as<const memory&>().count() * data_size);
    memset(        output_grad.as<const memory&>().pointer, 0,         output_grad.as<const memory&>().count() * data_size);
    memset(       current_mean.as<const memory&>().pointer, 0,        current_mean.as<const memory&>().count() * data_size);
    memset(current_inv_std_dev.as<const memory&>().pointer, 0, current_inv_std_dev.as<const memory&>().count() * data_size);

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
    auto data_size = memory::traits(input.as<const memory&>().argument.format).type->size;
    memset(      input.as<const memory&>().pointer, 0,       input.as<const memory&>().count() * data_size);
    memset(      scale.as<const memory&>().pointer, 0,       scale.as<const memory&>().count() * data_size);
    memset(       bias.as<const memory&>().pointer, 0,        bias.as<const memory&>().count() * data_size);
    memset(    average.as<const memory&>().pointer, 0,     average.as<const memory&>().count() * data_size);
    memset(inv_std_dev.as<const memory&>().pointer, 0, inv_std_dev.as<const memory&>().count() * data_size);

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
    auto data_size = memory::traits(input.as<const memory&>().argument.format).type->size;
    memset(      input.as<const memory&>().pointer, 0,       input.as<const memory&>().count() * data_size);
    memset(      scale.as<const memory&>().pointer, 0,       scale.as<const memory&>().count() * data_size);
    memset(       bias.as<const memory&>().pointer, 0,        bias.as<const memory&>().count() * data_size);
    memset(    average.as<const memory&>().pointer, 0,     average.as<const memory&>().count() * data_size);
    memset(inv_std_dev.as<const memory&>().pointer, 0, inv_std_dev.as<const memory&>().count() * data_size);

    // Create primitive.
    auto bn = batch_normalization_inference::create({engine::reference, {output}, {input, scale, bias, average, inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_complex_example_training_float() 
{
    // Create data buffers.
    // TODO: after string keys have been implemented, move buffer creation to primitives' constructors and reference them by primitive-string lookups.
    auto forward_input       = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto forward_scale       = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto forward_bias        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto current_mean        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto moving_mean         = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto moving_inv_std_dev  = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto forward_output      = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    //auto forward_output_grad = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true}); // same as forward_output
    auto forward_input_grad  = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto forward_scale_grad  = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto forward_bias_grad   = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    // Create primitives.
    auto bn_train_fw  = batch_normalization_training_forward::create({engine::reference, {forward_output, current_mean, current_inv_std_dev, moving_mean, moving_inv_std_dev}, {forward_input, forward_scale, forward_bias}, true, 0.0, FLT_EPSILON});
    auto bn_train_bck = batch_normalization_training_backward::create({engine::reference, {forward_input_grad, forward_scale_grad, forward_bias_grad}, {forward_input, forward_scale, forward_bias, forward_output, current_mean, current_inv_std_dev}, true});
    auto bn_infer     = batch_normalization_inference::create({engine::reference, {forward_output}, {forward_input, forward_scale, forward_bias, moving_mean, moving_inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
    {
        execute({bn_train_fw});

        // TODO: add some simple error function instead of directly connecting forward and backward pass by forward_output primitive
        // ....

        execute({bn_train_bck});

        // TODO: add SGD primitive
        // ...

        // TODO: after above is completed, whole training stack should be done via single execute() call.
    }

    // Run few times.
    for(int i = 0; i < 3; ++i)
    {
        execute({bn_infer});
    }
}

void spatial_bn_complex_example_training_double() 
{
    // Create data buffers.
    // TODO: after string keys have been implemented, move buffer creation to primitives' constructors and reference them by primitive-string lookups.
    auto forward_input       = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto forward_scale       = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto forward_bias        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto current_mean        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto moving_mean         = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto moving_inv_std_dev  = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto forward_output      = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    //auto forward_output_grad = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true}); // same as forward_output
    auto forward_input_grad  = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto forward_scale_grad  = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto forward_bias_grad   = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Create primitives.
    auto bn_train_fw  = batch_normalization_training_forward::create({engine::reference, {forward_output, current_mean, current_inv_std_dev, moving_mean, moving_inv_std_dev}, {forward_input, forward_scale, forward_bias}, true, 0.0, FLT_EPSILON});
    auto bn_train_bck = batch_normalization_training_backward::create({engine::reference, {forward_input_grad, forward_scale_grad, forward_bias_grad}, {forward_input, forward_scale, forward_bias, forward_output, current_mean, current_inv_std_dev}, true});
    auto bn_infer     = batch_normalization_inference::create({engine::reference, {forward_output}, {forward_input, forward_scale, forward_bias, moving_mean, moving_inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
    {
        execute({bn_train_fw});

        // TODO: add some simple error function instead of directly connecting forward and backward pass by forward_output primitive
        // ....

        execute({bn_train_bck});

        // TODO: SGD
        // ...

        // TODO: after above is completed, whole training stack should be done via single execute() call.
    }

    // Run few times.
    for(int i = 0; i < 3; ++i)
    {
        execute({bn_infer});
    }
}