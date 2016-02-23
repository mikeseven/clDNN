#include "neural.h"

// memory->memory max pooling
void example_008() {
    using namespace neural;
    float *data_buffer = nullptr;

    const uint32_t output_y = 0,    // size of whole output buffer
                   output_x = 0,
                   output_z = 0,
                   output_b = 0,

                   input_y = 0,     // size of whole input buffer
                   input_x = 0,
                   input_z = 0,
                   input_b = 0,

                   out_off_y = 0,
                   out_off_x = 0,
                   out_off_z = 0,
                   out_off_b = 0,

                   out_siz_y = 0,   // size of area to do pooling after offset
                   out_siz_x = 0,
                   out_siz_z = 0,
                   out_siz_b = 0,

                   stride_y = 0,
                   stride_x = 0,
                   stride_z = 0,
                   stride_b = 0,

                   pooling_siz_y = 1,   // size of pooling window
                   pooling_siz_x = 1,
                   pooling_siz_z = 1,
                   pooling_siz_b = 1;

     const int32_t in_off_y = 0,
                   in_off_x = 0,
                   in_off_z = 0,
                   in_off_b = 0;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {16, 32, 64, 128}});

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

    execute({input(data_buffer), output(data_buffer), act});
}
