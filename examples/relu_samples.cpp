#include "api/neural.h"

// memory->memory relu
void example_relu_forward() {
    using namespace neural;
    float *data_buffer = nullptr;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});

    auto act    = relu::create({engine::reference, output, input});

    execute({input(data_buffer), output(data_buffer), act});
}

void example_relu_backward() {
    using namespace neural;

    auto forward_input  = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});
    auto forward_output_grad = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});

    auto forward_input_grad = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}, true});

    auto act    = relu_backward::create({engine::reference, {forward_input_grad}, {forward_output_grad, forward_input}});

    execute({act});
}
