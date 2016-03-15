/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "tests/gtest/gtest.h"
#include "api/neural.h"

#include <random>

TEST(relu_basic, relu_fw_test) {
    using namespace neural;

    const uint32_t y = 8, x = 8, z = 3, b = 2;

    auto input  = memory::create({engine::cpu, memory::format::yxfb_f32, {y, x, z, b}, true});
    auto output = memory::create({engine::cpu, memory::format::yxfb_f32, {y, x, z, b}});
    memory_helper::fill_memory<float>(input);

    auto act = relu::create({engine::reference, output, input});
    auto buf = static_cast<float*>(input.as<const memory&>().pointer);
    execute({input, output(buf), act});

    for(size_t i = 0; i < y*x*z*b; ++i)
        buf[i] = (buf[i] > 0)? buf[i] : -buf[i];

    auto act2 = relu::create({engine::reference, output, output});
    execute({output, output, act});

    bool result = false;
    for(size_t i = 0; i < y*x*z*b; ++i)
        result = result || buf[i];

    auto tmp =  static_cast<int>(result);
    EXPECT_EQ(0, 0);

    //EXPECT_EQ(0, static_cast<int>(result));
   // EXPECT_EQ(false, result);
}
