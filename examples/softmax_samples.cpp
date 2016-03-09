#include "api/neural.h"

// memory->memory softmax
void example_softmax_forward() {
    using namespace neural;

    const uint32_t output_x  = 7, output_b  = 3,  // size of whole output buffer
                   input_x   = 6, input_b   = 2,  // size of whole input buffer
                   out_off_x = 0, out_off_b = 1,
                   out_siz_x = 5, out_siz_b = 2;  // size of area to do softmax after offset

    const int32_t  in_off_x  = 1, in_off_b  = 0;

    float in_buffer[input_x*input_b];
    float out_buffer[output_x*output_b];
    // input buffer should be initialized with valid data

    auto input  = memory::create({engine::cpu, memory::format::xb_f32, {input_x, input_b}});
    auto output = memory::create({engine::cpu, memory::format::xb_f32, {output_x, output_b}});

    auto act    = normalization::softmax::create( {engine::reference,
                                                   output,
                                                   {out_off_x, out_off_b},
                                                   {out_siz_x, out_siz_b},
                                                   input,
                                                   {in_off_x, in_off_b}
                                                  });

    execute({input(in_buffer), output(out_buffer), act});
}

void example_softmax_backward(){
    // todo softmax bw
}