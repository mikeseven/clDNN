/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "tests/gtest/gtest.h"
#include "api/neural.h"
#include "multidimensional_counter.h"
#include <iostream> // todo remove
using namespace neural;

TEST(fully_connected, basic) {
    
    const uint32_t output_x  = 4, output_b  = 1,  // size of whole output buffer
                   input_x   = 3, input_b   = 1,  // size of whole input buffer
                   weight_x  = 4, weight_y  = 3;  // size of whole weights buffer

    auto input_prim   = memory::create({ engine::cpu, memory::format::x_f32,{ input_x,  input_b  } , true });
    auto output_prim  = memory::create({ engine::cpu, memory::format::xb_f32,{ output_x, output_b } , true });
    auto weights_prim = memory::create({ engine::cpu, memory::format::xb_f32,{ weight_x, weight_y } , true });
    auto full_con_prim = fully_connected::create({ engine::reference, output_prim, input_prim, weights_prim });

    auto& input_memory   = input_prim.as<const memory&>();
    auto& output_memory  = output_prim.as<const memory&>();
    auto& weights_memory = weights_prim.as<const memory&>();

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  2.0f);
    input_memory.set_value<float>(2,  0.5f);

    weights_memory.set_value<float>(0,  1.5f);
    weights_memory.set_value<float>(1,  1.0f);
    weights_memory.set_value<float>(2,  0.5f);
    weights_memory.set_value<float>(3, -1.0f);
    weights_memory.set_value<float>(4,  0.0f);
    weights_memory.set_value<float>(5,  0.5f);
    weights_memory.set_value<float>(6,  0.5f);
    weights_memory.set_value<float>(7, -0.5f);
    weights_memory.set_value<float>(8, -2.0f);
    weights_memory.set_value<float>(9, -0.5f);
    weights_memory.set_value<float>(10, 1.0f);
    weights_memory.set_value<float>(11, 1.5f);

    output_memory.fill<float>(0.0f);

    try {
        execute({full_con_prim});
    }
    catch (std::exception &e) {
        std::cout << e.what();
    }

    EXPECT_EQ(1.5f,   output_memory.get_value<float>(0));
    EXPECT_EQ(0.75f,  output_memory.get_value<float>(1));
    EXPECT_EQ(-2.25f, output_memory.get_value<float>(2));
    EXPECT_EQ(3.0f,   output_memory.get_value<float>(3));
}