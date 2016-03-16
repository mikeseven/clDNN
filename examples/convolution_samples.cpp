#include "api/neural.h"

// memory->memory convolution
void example_convolution_forward() {
    using namespace neural;

    const uint32_t output_y    = 7, output_x    = 7, output_z    = 3, output_b    = 3,  // size of whole output buffer
                   input_y     = 6, input_x     = 6, input_z     = 2, input_b     = 2,  // size of whole input buffer
                   out_off_y   = 1, out_off_x   = 2, out_off_z   = 1, out_off_b   = 0,
                   out_siz_y   = 5, out_siz_x   = 5, out_siz_z   = 2, out_siz_b   = 2,  // size of area to do convolution after offset
                   stride_y    = 1, stride_x    = 1, stride_z    = 1, stride_b    = 1,
                   conv_size_y = 2, conv_size_x = 2, conv_size_z = 1, conv_size_b = 1;  // size of convolution window

    const int32_t in_off_y = 0, in_off_x = 0, in_off_z = 0, in_off_b = 0;

    float in_buffer[input_y*input_x*input_z*input_b];
    float out_buffer[output_y*output_x*output_z*output_b];
    float weight_buffer[conv_size_y*conv_size_x*conv_size_z*conv_size_b];
    float bias_buffer[out_siz_z];
    // buffers should be initialized with valid data

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {input_y, input_x, input_z, input_b}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {output_y, output_x, output_z, output_b}});
    auto weights= memory::create({engine::cpu, memory::format::yxfb_f32, {conv_size_y, conv_size_x, conv_size_z, conv_size_b}});
    auto biases = memory::create({engine::cpu, memory::format::   x_f32, {out_siz_z}});

    auto act    = convolution::create( {engine::reference,
                                        output,
                                        {out_off_y, out_off_x, out_off_z, out_off_b},
                                        {out_siz_y, out_siz_x, out_siz_z, out_siz_b},
                                        input,
                                        {in_off_y, in_off_x, in_off_z, in_off_b},
                                        {stride_y, stride_x, stride_z, stride_b},
                                        weights,
                                        biases,
                                        padding::zero}
                                      );

    execute({input(in_buffer), output(out_buffer), weights(weight_buffer), biases(bias_buffer), act});
}

void example_convolution_backward(){
    using namespace neural;

    const uint32_t output_y    = 2,
                   output_x    = 2,
                   output_z    = 1,
                   output_b    = 1, // size of whole output buffer

                   input_y     = 3,
                   input_x     = 3,
                   input_z     = 1,
                   input_b     = 1,  // size of whole input buffer
        /*
                   out_off_y   = 1,
                   out_off_x   = 2,
                   out_off_z   = 1,
                   out_off_b   = 0,

                   out_siz_y   = 5,
                   out_siz_x   = 5,
                   out_siz_z   = 2,
                   out_siz_b   = 2,  // size of area to do convolution after offset
        */
                   out_siz_z   = 1,

                   stride_y    = 1,
                   stride_x    = 1,
                   stride_z    = 1,
                   stride_b    = 1,

                   conv_size_y = 2,
                   conv_size_x = 2,
                   conv_size_z = 1,
                   conv_size_b = 1;  // size of convolution window

    const int32_t in_off_y = 0,
                  in_off_x = 0,
                  in_off_z = 0,
                  in_off_b = 0;

    auto bw_output    = memory::create({engine::cpu, memory::format::yxfb_f32, {output_y, output_x, output_z, output_b}, true});
    auto bw_input     = memory::create({engine::cpu, memory::format::yxfb_f32, {input_y, input_x, input_z, input_b}, true});
    auto fw_input     = memory::create({engine::cpu, memory::format::yxfb_f32, {output_y, output_x, output_z, output_b}, true});
    auto weights      = memory::create({engine::cpu, memory::format::yxfb_f32, {conv_size_y, conv_size_x, conv_size_z, conv_size_b}, true});
    auto weights_diff = memory::create({engine::cpu, memory::format::yxfb_f32, {conv_size_y, conv_size_x, conv_size_z, conv_size_b}, true});
    auto biases       = memory::create({engine::cpu, memory::format::x_f32,    {out_siz_z}, true});
    auto biases_diff  = memory::create({engine::cpu, memory::format::x_f32,    {out_siz_z}, true});
    // buffers should be initialized with valid data
    // *diff buffers must be filled with '0'

    auto act = convolution_backward::create({ engine::reference,
                                              std::vector<primitive>({bw_output, weights_diff, biases_diff}),
                                          //   {out_off_y, out_off_x, out_off_z, out_off_b},
                                          //   {out_siz_y, out_siz_x, out_siz_z, out_siz_b},
                                              {bw_input, fw_input, weights, biases},
                                          //  {in_off_y, in_off_x, in_off_z, in_off_b},
                                              {stride_y, stride_x, stride_z, stride_b},
                                              padding::zero
                                            });

    execute({
        bw_input, fw_input, weights, biases,  //inputs
        bw_output, weights_diff, biases_diff, //outputs
        act
    });
}
