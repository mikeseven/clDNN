#include "neural.h"

// memory->memory convolution with weights & biases from file
void example_000() {
    char *data_buffer = nullptr;
    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::xyzb, {224, 224, 3,  24}});
    auto output = memory::create({engine::cpu, memory::format::xyzb, {112, 112, 96, 24}});
    auto weight = file::create({engine::cpu, "weight.nnb"});
    auto bias   = file::create({engine::cpu, "bias.nnb"});

    auto conv  = convolution::create({engine::cpu, output, input, weight, bias, padding::zero});

    execute({input(data_buffer), output(data_buffer), conv});
}