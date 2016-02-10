#include "neural.h"
#include "thread_pool.h"

#include <random>
static float frand(){
    float f;
    UINT32 *fi = (UINT32*)&f;
    *fi = 0;
    const int minBitsRandGives  = (1<<15);          //  RAND_MAX is at least (1<<15)    
    UINT32 randExp              = (rand()%254)+1;   //  Exponents are in range of [1..254]
    UINT32 randMantissa         = ((rand() % minBitsRandGives) << 8) | (rand()%256);
    *fi                         = randMantissa | (randExp<<23);                 // Build a float with random exponent and random mantissa
    return f;
}

void foo() {
    char *data_buffer = nullptr;
    using namespace neural;


const int size = 192;
    float input_buf [size];
    float output_buf[size];
    
    for(int i = 0; i < size ; ++i)
        input_buf[i] = frand();

    auto input  = memory::create({engine::cpu, memory::format::xyzb, {4, 4, 3, 2}});
    auto output = memory::create({engine::cpu, memory::format::xyzb, {4, 4, 3, 2}});

 //   auto input  = memory::create({engine::cpu, memory::format::xyzb, {16, 32, 64, 128}});
 //   auto output = memory::create({engine::cpu, memory::format::xyzb, {16, 32, 64, 128}});

    auto act    = relu::create({engine::reference, output, input});

    act.input[0];
    act.input.size();
    act.output[0];
    act.output.size();

    act.as<const relu &>().argument.engine;
    act.as<const relu &>().argument.input[0];
    act.as<const relu &>().argument.input.size();

    execute({input(input_buf), output(output_buf), act});
     //   execute({input(data_buffer), output(data_buffer), act});
}

auto main(int, char *[]) -> int {
    using namespace neural;

    char  *input_buffer = nullptr;
    char *output_buffer = nullptr;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {224, 224, 3,  24}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {224, 224, 96, 24}});
    auto weight = file::create({engine::cpu, "weight.nnb"});
    auto bias   = file::create({engine::cpu, "bias.nnb"});

    auto conv  = convolution::create({engine::cpu, output, input, weight, bias, padding::zero});

    conv.as<const convolution &>().argument.input;

    execute({input(input_buffer), output(output_buffer), conv});
}