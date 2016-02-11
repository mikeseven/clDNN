#include "neural.h"
#include "thread_pool.h"


void example() {
    using namespace neural;
    float data_buffer[120];

    for(unsigned i = 0; i < 120; ++i )
        data_buffer[i] = i - 8;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 3, 4, 5}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 3, 4, 5}});

    auto act    = relu::create({engine::reference, output, {0,0,0,0}, {2, 3, 4, 5}, input, {0,0,0,0} });

    execute({input(data_buffer), output(data_buffer), act});
}

auto main(int, char *[]) -> int {
    using namespace neural;

    char  *input_buffer = nullptr;
    char *output_buffer = nullptr;

    example();
    //auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {224, 224, 3,  24}});
    //auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {224, 224, 96, 24}});
    //auto weight = file::create({engine::cpu, "weight.nnb"});
    //auto bias   = file::create({engine::cpu, "bias.nnb"});

    //auto conv  = convolution::create({engine::cpu, output, input, weight, bias, padding::zero});

    //conv.as<const convolution &>().argument.input;

    //execute({input(input_buffer), output(output_buffer), conv});
}