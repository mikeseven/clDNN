#include "neural.h"
#include <iostream>
#include <algorithm>

// choose fastest convolution & use it
void example_006() {
    using namespace neural;

    char  *input_buffer = nullptr;
    char *output_buffer = nullptr;

    auto input  = memory::create({engine::cpu, memory::format::xyzb, {224, 224, 3,  24}});
    auto output = memory::create({engine::cpu, memory::format::xyzb, {112, 112, 96, 24}});
    auto weight = file::create({engine::cpu, "weight.nnb"});
    auto bias   = file::create({engine::cpu, "bias.nnb"});

    auto result = convolution::query({engine::any, memory::format::any, {}, {}, input, {-1,-1}, {1, 1}, weight, bias, padding::zero});
    for(auto &entry : result)
        std::cout << entry["engine"].u64() << ":" << entry["engine"].s() << " = " << entry["time"].f32()*1000.0f << "ms" << std::endl;

    // sort to find out the fastest
    auto compare_less_time = [](is_a_query_entry a, is_a_query_entry b) {return a["time"].f32()<b["time"].f32();};
    auto best_time_arg = std::min_element(result.begin(), result.end(), compare_less_time)->arguments;
    primitive conv  = convolution::create(best_time_arg);

    execute({input(input_buffer), output(input_buffer), conv});
}
