/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "tests/gtest/gtest.h"
#include "api/neural.h"

using namespace neural;

TEST(pooling_forward, max_yxfb_f32_wsiz3x3_wstr1x1_i3x3x1x1_nopad) {
    auto input_prim  = memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto output_prim = memory::create({engine::cpu, memory::format::yxfb_f32, {1, 1, 1, 1}, true});
    auto pool_prim = pooling::create({engine::reference, pooling::mode::max, output_prim, input_prim, 1, 3, padding::type::zero});

    auto& input_memory = input_prim.as<const memory&>();
    auto& output_memory = output_prim.as<const memory&>();

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  1.0f);
    input_memory.set_value<float>(2,  0.5f);
    input_memory.set_value<float>(3,  2.0f);
    input_memory.set_value<float>(4,  1.5f);
    input_memory.set_value<float>(5, -0.5f);
    input_memory.set_value<float>(6,  0.0f);
    input_memory.set_value<float>(7, -1.0f);
    input_memory.set_value<float>(8,  0.5f);

    output_memory.fill<float>(0.0f);

    execute({pool_prim});

    EXPECT_EQ(2.0f, output_memory.get_value<float>(0));
}

TEST(pooling_forward, max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x1x1_nopad) {
    auto input_prim  = memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 1, 1}, true});
    auto output_prim = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto pool_prim = pooling::create({engine::reference, pooling::mode::max, output_prim, input_prim, 1, 2, padding::type::zero});

    auto& input_memory = input_prim.as<const memory&>();
    auto& output_memory = output_prim.as<const memory&>();

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  1.0f);
    input_memory.set_value<float>(2,  0.5f);
    input_memory.set_value<float>(3,  2.0f);
    input_memory.set_value<float>(4,  1.5f);
    input_memory.set_value<float>(5, -0.5f);
    input_memory.set_value<float>(6,  0.0f);
    input_memory.set_value<float>(7, -1.0f);
    input_memory.set_value<float>(8,  0.5f);

    output_memory.fill<float>(0.0f);

    execute({pool_prim});

    EXPECT_EQ(2.0f, output_memory.get_value<float>(0));
    EXPECT_EQ(1.5f, output_memory.get_value<float>(1));
    EXPECT_EQ(2.0f, output_memory.get_value<float>(2));
    EXPECT_EQ(1.5f, output_memory.get_value<float>(3));
}

TEST(pooling_forward, max_yxfb_f32_wsiz2x2_wstr2x2_i4x4x1x1_nopad) {
    auto input_prim  = memory::create({engine::cpu, memory::format::yxfb_f32, {4, 4, 1, 1}, true});
    auto output_prim = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto pool_prim = pooling::create({engine::reference, pooling::mode::max, output_prim, input_prim, 2, 2, padding::type::zero});

    auto& input_memory = input_prim.as<const memory&>();
    auto& output_memory = output_prim.as<const memory&>();

    input_memory.set_value<float>( 0, -0.25f);
    input_memory.set_value<float>( 1,  1.00f);
    input_memory.set_value<float>( 2,  0.50f);
    input_memory.set_value<float>( 3,  0.25f);
    input_memory.set_value<float>( 4,  2.00f);
    input_memory.set_value<float>( 5,  1.50f);
    input_memory.set_value<float>( 6, -0.50f);
    input_memory.set_value<float>( 7, -0.75f);
    input_memory.set_value<float>( 8,  0.00f);
    input_memory.set_value<float>( 9, -1.00f);
    input_memory.set_value<float>(10,  0.50f);
    input_memory.set_value<float>(11,  0.25f);
    input_memory.set_value<float>(12,  0.50f);
    input_memory.set_value<float>(13, -2.00f);
    input_memory.set_value<float>(14, -1.50f);
    input_memory.set_value<float>(15, -2.50f);

    output_memory.fill<float>(0.0f);

    execute({pool_prim});

    EXPECT_EQ(2.0f, output_memory.get_value<float>(0));
    EXPECT_EQ(0.5f, output_memory.get_value<float>(1));
    EXPECT_EQ(0.5f, output_memory.get_value<float>(2));
    EXPECT_EQ(0.5f, output_memory.get_value<float>(3));
}