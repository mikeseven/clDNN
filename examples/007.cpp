#include "neural.h"

// memory->memory relu
void example_007() {
    using namespace neural;
    #define X 32
    #define Y 16
    #define Z 64
    #define B 128
    const int size = X*Y*Z*B;
    //float input_buf [size];
    //float output_buf[size];

    float *data_buffer = new float[size];
    for(int i = 0; i < size; ++i){
        data_buffer[i] = i - size/2;
    }

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});

    auto act    = relu::create({engine::reference, output, input});

    act.input[0];
    act.input.size();
    act.output[0];
    act.output.size();

    act.as<const relu &>().argument.engine;
    act.as<const relu &>().argument.input[0];
    act.as<const relu &>().argument.input.size();

   execute({input(data_buffer), output(data_buffer), act});
}
