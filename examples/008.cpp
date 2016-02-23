#include "neural.h"

// memory->memory max pooling
void example_008() {
    using namespace neural;

    const uint32_t output_y = 2,    // size of whole output buffer
                   output_x = 2,
                   output_z = 1,
                   output_b = 1,

                   input_y = 3,     // size of whole input buffer
                   input_x = 3,
                   input_z = 1,
                   input_b = 1,

                   out_off_y = 0,
                   out_off_x = 0,
                   out_off_z = 0,
                   out_off_b = 0,

                   out_siz_y = 2,   // size of area to do pooling after offset
                   out_siz_x = 2,
                   out_siz_z = 1,
                   out_siz_b = 1,

                   stride_y = 1,
                   stride_x = 1,
                   stride_z = 0,
                   stride_b = 0,

                   pooling_siz_y = 2,   // size of pooling window
                   pooling_siz_x = 2,
                   pooling_siz_z = 1,
                   pooling_siz_b = 1;

     const int32_t in_off_y = 0,
                   in_off_x = 0,
                   in_off_z = 0,
                   in_off_b = 0;

    float in_buffer[input_y*input_x*input_z*input_b];
    float out_buffer[output_y*output_x*output_z*output_b];

    for(int i = 0; i < input_y*input_x*input_z*input_b; ++i )
        in_buffer[i] = 1.0f * i - 1.0f * input_y*input_x*input_z*input_b/2;

    for(int i = 0; i < output_y*output_x*output_z*output_b; ++i )
        out_buffer[i] = -999;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {input_y, input_x, input_z, input_b}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {output_y, output_x, output_z, output_b}});

    auto act    = pooling::create( {engine::reference,
                                    pooling::mode::max,
                                    memory::format::yxfb_f32,
                                    {out_off_y, out_off_x, out_off_z, out_off_b},
                                    {out_siz_y, out_siz_x, out_siz_z, out_siz_b},
                                    input,
                                    {in_off_y, in_off_x, in_off_z, in_off_b},
                                    {stride_y, stride_x, stride_z, stride_b},
                                    {pooling_siz_y, pooling_siz_x, pooling_siz_z, pooling_siz_b},
                                    padding::zero}
                                  );

    execute({input(in_buffer), output(out_buffer), act});
}
