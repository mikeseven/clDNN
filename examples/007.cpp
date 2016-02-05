#include "neural.h"

// memory->memory relu
void example_007() {
    char *data_buffer = nullptr;
    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::xyzb, {16, 32, 64, 128}});
    auto output = memory::create({engine::cpu, memory::format::xyzb, {16, 32, 64, 128}});

    auto act  = relu::create({engine::reference, output, input});

    act.input[0];
    act.input.size();
    act.output[0];
    act.output.size();

    act.as<const relu &>().argument.engine;
    act.as<const relu &>().argument.input[0];
    act.as<const relu &>().argument.input.size();

    execute({input(data_buffer), output(data_buffer), act});
}