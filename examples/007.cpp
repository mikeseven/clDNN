#include "neural.h"

// memory->memory relu
void example_007() {
    using namespace neural;
    float *data_buffer = nullptr;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});

    auto act    = relu::create({engine::reference, output, input});

    execute({input(data_buffer), output(data_buffer), act});
}
