#if 0
#include "api/neural.h"

// memory->memory convolution with weights & biases from file
void example_000() {
    char *in_ptr = nullptr, *out_ptr = nullptr;
    using namespace neural;
    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {224, 224, 3,  24}});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {224, 224, 96, 24}});
    auto weight = file::create({engine::cpu, "weight.nnb"});
    auto bias   = file::create({engine::cpu, "bias.nnb"});
    auto conv  = convolution::create({engine::cpu, output, input, weight, bias, padding::zero});

    conv["engine"].s();     // std::string("cpu")
    conv["engine"].u64();   // uint64_t(1)
    conv["time"].f32();     // float(0.000001f)
    conv["inputs"].u32();   // uint32_t(3)
    conv["input0"].s();     // std::string("input")
    conv["input1"].s();     // std::string("weight")
    conv["input2"].s();     // std::string("bias")
    conv["name"].s();       // std::string("convolution")
    conv["info_short"].s(); // std::string("direct convolution")

    execute({input(in_ptr), output(out_ptr), conv});
}
#endif