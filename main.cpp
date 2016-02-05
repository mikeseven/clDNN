#include "neural.h"
#include "thread_pool.h"

auto main(int, char *[]) -> int {
    using namespace neural;

    char  *input_buffer = nullptr;
    char *output_buffer = nullptr;

    auto input  = memory::create({engine::cpu, memory::format::xyzb, {224, 224, 3,  24}});
    auto output = memory::create({engine::cpu, memory::format::xyzb, {112, 112, 96, 24}});
    auto weight = file::create({engine::cpu, "weight.nnb"});
    auto bias   = file::create({engine::cpu, "bias.nnb"});

    auto conv  = convolution::create({engine::cpu, output, input, weight, bias, padding::zero});

    conv.as<const convolution *>()->argument.input;

    execute({input(input_buffer), output(output_buffer), conv});
}