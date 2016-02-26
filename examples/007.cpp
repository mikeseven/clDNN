#include "neural.h"

// memory->memory relu
void example_007() {
using namespace neural;
    const int  in_offset_x = 3,
               in_offset_y = 3,
               in_offset_z = 1,
               in_offset_b = 2;
    const unsigned in_x = 6,
                   in_y = 6,
                   in_z = 2,
                   in_b = 3,
                   out_x = 8,
                   out_y = 8,
                   out_z = 2,
                   out_b = 3,
                   out_offset_x = 1,
                   out_offset_y = 1,
                   out_offset_z = 0,
                   out_offset_b = 0,
                   out_size_x = in_x - in_offset_x,
                   out_size_y = in_y - in_offset_y,
                   out_size_z = in_z - in_offset_z,
                   out_size_b = in_b - in_offset_b;

    float data_buffer[in_x*in_y*in_z*in_b];
    float out_buffer[out_x*out_y*out_z*out_b];

    for(int i = 0; i < in_x*in_y*in_z*in_b; ++i )
        data_buffer[i] = static_cast<float>(i - static_cast<int>(in_x*in_y*in_z*in_b)/2);

    for(int i = 0; i < out_x*out_y*out_z*out_b; ++i )
        out_buffer[i] = -999;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {in_x, in_y, in_z, in_b}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {out_x, out_y, out_z, out_b}});

    auto act    = relu::create({engine::reference, output, {out_offset_x, out_offset_y, out_offset_z, out_offset_b}, {out_size_x, out_size_y, out_size_z, out_size_b}, input, {in_offset_x, in_offset_y, in_offset_z, in_offset_b}} );

    execute({input(data_buffer), output(out_buffer), act});
}
