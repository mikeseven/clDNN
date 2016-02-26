#if 0
#include "api/neural.h"

// convolution->relu->pooling->lrn with weights & biases from file
void example_003() {
    char *data_buffer = nullptr;
    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {224, 224, 3,  24}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {112, 112, 96, 24}});
    auto weight = file::create({engine::cpu, "weight.nnb"});
    auto bias   = file::create({engine::cpu, "bias.nnb"});

    auto conv   = convolution::create({engine::cpu, memory::format::yxfb_f32, input, weight, bias, padding::zero});
    auto act    = relu::create({engine::cpu, memory::format::yxfb_f32, conv});
    auto pool   = pooling::create({engine::cpu, pooling::mode::max, memory::format::yxfb_f32, act, 3, 2, padding::zero});
    auto lrn    = normalization::response::create({engine::cpu, output, pool, 5, padding::zero, 1.0f, 0.00002f, 0.75f });

    execute({input(data_buffer), output(data_buffer), conv, act, pool, lrn});
}
#endif

