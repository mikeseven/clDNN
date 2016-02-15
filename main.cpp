#include "neural.h"
#include "thread_pool.h"

void example() {
    using namespace neural;
    const unsigned in_x = 6,
                   in_y = 6,
                   in_z = 2,
                   in_b = 2,
                   out_x = 6,
                   out_y = 6,
                   out_z = 2,
                   out_b = 2;
    const int  in_offset_x = 0,
               in_offset_y = 0,
               in_offset_z = 0,
               in_offset_b = 0;

    float data_buffer[in_x*in_y*in_z*in_b];
    float out_buffer[out_x*out_y*out_z*out_b];

    for(int i = 0; i < in_x*in_y*in_z*in_b; ++i )
        data_buffer[i] = static_cast<float>(i - static_cast<int>(in_x*in_y*in_z*in_b)/2);


    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {in_x, in_y, in_z, in_b}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {out_x, out_y, out_z, out_b}});
    
    
    auto act    = relu::create({engine::reference, output, {0,0,0,0}, {out_x, out_y, out_z, out_b}, input, {in_offset_x, in_offset_y, in_offset_z, in_offset_b}} );
    //auto act    = relu::create({engine::reference, output, {0,0,0,0}, {x, y, z, b}, input, {0,0,0,0} });

    execute({input(data_buffer), output(out_buffer), act});

    if(1)
        int x = 1;

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