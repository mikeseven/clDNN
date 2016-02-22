#include "neural.h"

using namespace neural;

// Some shady helpers.
namespace 
{
    template <class T> T* get_mem_first(const neural::primitive &mem) {return reinterpret_cast<T*>(mem.as<const memory&>().pointer);}
    template <class T> T* get_mem_last(const neural::primitive &mem) {return get_mem_first<T>(mem) + mem.as<const memory&>().count();}

    template <class T> 
    void fill_memory(neural::primitive &mem, T value)
    {
        if(type_id<T>()->id != memory::traits(mem.as<const memory&>().argument.format).type->id)
            throw std::runtime_error("fill_memory: types do not match");

        std::fill(get_mem_first<T>(mem), get_mem_last<T>(mem), value);
    }
}

void spatial_bn_trivial_example_forward_training_float() 
{
    // Create input buffers.
    auto input               = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto scale               = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto bias                = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    
    // Create output buffers.
    auto output              = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto current_average     = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto moving_average      = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto moving_inv_std_dev  = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    // Initialize input buffers.
    fill_memory<float>(input, 0);
    fill_memory<float>(scale, 0);
    fill_memory<float>( bias, 0);

    // Create primitive.
    auto bn = normalization::batch_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, true, 0.0, FLT_EPSILON});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_forward_training_double() 
{
    // Create input buffers.
    auto input               = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto scale               = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto bias                = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Create output buffers.
    auto output              = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto current_average     = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto moving_average      = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto moving_inv_std_dev  = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Initialize input buffers.
    fill_memory<double>(input, 0);
    fill_memory<double>(scale, 0);
    fill_memory<double>( bias, 0);

    // Create primitive.
    auto bn = normalization::batch_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, true, 0.0, DBL_EPSILON});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_backward_training_float() 
{
    // Create input buffers.
    auto forward_input       = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto forward_scale       = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto forward_bias        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto output_grad         = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto current_mean        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    // Create output buffers.
    auto input_grad = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto scale_grad = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto bias_grad  = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    // Initialize input buffers.
    fill_memory<float>(      forward_input, 0);
    fill_memory<float>(      forward_scale, 0);
    fill_memory<float>(       forward_bias, 0);
    fill_memory<float>(        output_grad, 0);
    fill_memory<float>(       current_mean, 0);
    fill_memory<float>(current_inv_std_dev, 0);

    // Create primitive.
    auto bn = normalization::batch_training_backward::create({engine::reference, {input_grad, scale_grad, bias_grad}, {forward_input, forward_scale, forward_bias, output_grad, current_mean, current_inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_backward_training_double() 
{
    // Create input buffers.
    auto forward_input       = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto forward_scale       = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto forward_bias        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto output_grad         = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto current_mean        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Create output buffers.
    auto input_grad = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto scale_grad = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto bias_grad  = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Initialize input buffers.
    fill_memory<double>(      forward_input, 0);
    fill_memory<double>(      forward_scale, 0);
    fill_memory<double>(       forward_bias, 0);
    fill_memory<double>(        output_grad, 0);
    fill_memory<double>(       current_mean, 0);
    fill_memory<double>(current_inv_std_dev, 0);

    // Create primitive.
    auto bn = normalization::batch_training_backward::create({engine::reference, {input_grad, scale_grad, bias_grad}, {forward_input, forward_scale, forward_bias, output_grad, current_mean, current_inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_inference_float() 
{
    // Create input buffers.
    auto input       = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto scale       = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto bias        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto average     = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    
    // Create output buffer.
    auto output      = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});

    // Initialize input buffers.
    fill_memory<float>(      input, 0);
    fill_memory<float>(      scale, 0);
    fill_memory<float>(       bias, 0);
    fill_memory<float>(    average, 0);
    fill_memory<float>(inv_std_dev, 0);

    // Create primitive.
    auto bn = normalization::batch_inference::create({engine::reference, {output}, {input, scale, bias, average, inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_trivial_example_inference_double() 
{
    // Create input buffers.
    auto input       = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto scale       = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto bias        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto average     = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Create output buffer.
    auto output      = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});

    // Initialize input buffers.
    fill_memory<double>(      input, 0);
    fill_memory<double>(      scale, 0);
    fill_memory<double>(       bias, 0);
    fill_memory<double>(    average, 0);
    fill_memory<double>(inv_std_dev, 0);

    // Create primitive.
    auto bn = normalization::batch_inference::create({engine::reference, {output}, {input, scale, bias, average, inv_std_dev}, true});

    // Run few times.
    for(int i = 0; i < 3; ++i)
        execute({bn});
}

void spatial_bn_complex_example_training_float() 
{
    // TODO: after string keys have been implemented, move buffer creation to primitives' constructors and reference them by primitive-string lookups.

    // Create data buffers that have to be initialized before training starts.
    auto forward_input       = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto forward_scale       = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto forward_bias        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    // Create intermediate buffers that will be computed during forward training pass.
    auto current_mean        = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto moving_mean         = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto moving_inv_std_dev  = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    // Create output buffers.
    auto forward_output      = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    //auto forward_output_grad = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true}); // same as forward_output
    auto forward_input_grad  = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto forward_scale_grad  = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});
    auto forward_bias_grad   = memory::create({engine::cpu, memory::format::yxfb_f32, { 1,  1, 64,   1}, true});

    // Initialize input buffers for forward training primitive which will initialize other buffers.
    fill_memory<float>(forward_input, 0);
    fill_memory<float>(forward_scale, 0);
    fill_memory<float>( forward_bias, 0);

    // Create primitives.
    auto bn_train_fw  = normalization::batch_training_forward::create({engine::reference, {forward_output, current_mean, current_inv_std_dev, moving_mean, moving_inv_std_dev}, {forward_input, forward_scale, forward_bias}, true, 0.0, FLT_EPSILON});
    auto bn_train_bck = normalization::batch_training_backward::create({engine::reference, {forward_input_grad, forward_scale_grad, forward_bias_grad}, {forward_input, forward_scale, forward_bias, forward_output, current_mean, current_inv_std_dev}, true});
    auto bn_infer     = normalization::batch_inference::create({engine::reference, {forward_output}, {forward_input, forward_scale, forward_bias, moving_mean, moving_inv_std_dev}, true});

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
    // TODO: after string keys have been implemented, move buffer creation to primitives' constructors and reference them by primitive-string lookups.

    // Create data buffers that have to be initialized before training starts.
    auto forward_input       = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto forward_scale       = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto forward_bias        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Create intermediate buffers that will be computed during forward training pass.
    auto current_mean        = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto current_inv_std_dev = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto moving_mean         = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto moving_inv_std_dev  = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Create output buffers.
    auto forward_output      = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    //auto forward_output_grad = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true}); // same as forward_output
    auto forward_input_grad  = memory::create({engine::cpu, memory::format::yxfb_f64, {16, 32, 64, 128}, true});
    auto forward_scale_grad  = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});
    auto forward_bias_grad   = memory::create({engine::cpu, memory::format::yxfb_f64, { 1,  1, 64,   1}, true});

    // Initialize input buffers for forward training primitive which will initialize other buffers.
    fill_memory<double>(forward_input, 0);
    fill_memory<double>(forward_scale, 0);
    fill_memory<double>( forward_bias, 0);

    // Create primitives.
    auto bn_train_fw  = normalization::batch_training_forward::create({engine::reference, {forward_output, current_mean, current_inv_std_dev, moving_mean, moving_inv_std_dev}, {forward_input, forward_scale, forward_bias}, true, 0.0, FLT_EPSILON});
    auto bn_train_bck = normalization::batch_training_backward::create({engine::reference, {forward_input_grad, forward_scale_grad, forward_bias_grad}, {forward_input, forward_scale, forward_bias, forward_output, current_mean, current_inv_std_dev}, true});
    auto bn_infer     = normalization::batch_inference::create({engine::reference, {forward_output}, {forward_input, forward_scale, forward_bias, moving_mean, moving_inv_std_dev}, true});

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