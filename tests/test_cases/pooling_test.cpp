/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "tests/gtest/gtest.h"
#include "api/neural.h"

using namespace neural;

TEST(pooling_forward, basic_max_yxfb_f32_wsiz3x3_wstr1x1_i3x3x1x1_nopad) {
    /*  Brief test description.

        Pool window: 3x3
        Pool stride: 1x1
        Pool mode: max
        Padding: none

        Input data:
        [-0.5,  1.0,  0.5]
        [ 2.0,  1.5, -0.5]
        [ 0.0, -1.0,  0.5]

        Expected output:
        [ 2.0] 
    */

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

TEST(pooling_forward, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x1x1_nopad) {
    /*  Brief test description.

        Pool window: 2x2
        Pool stride: 1x1
        Pool mode: max
        Padding: none

        Input data:
        [-0.5,  1.0,  0.5]
        [ 2.0,  1.5, -0.5]
        [ 0.0, -1.0,  0.5]

        Expected output:
        [ 2.0,  1.5]
        [ 2.0,  1.5]
    */

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

TEST(pooling_forward, basic_max_yxfb_f32_wsiz2x2_wstr2x2_i4x4x1x1_nopad) {
    /*  Brief test description.

        Pool window: 2x2
        Pool stride: 2x2
        Pool mode: max
        Padding: none

        Input data:
        [-0.25,  1.00,  0.50,  0.25]
        [ 2.00,  1.50, -0.50, -0.75]
        [ 0.00, -1.00,  0.50,  0.25]
        [ 0.50, -2.00, -1.50, -2.50]

        Expected output:
        [ 2.0,  0.5]
        [ 0.5,  0.5]
    */

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

TEST(pooling_forward, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x2x2_nopad) {
    /*  Brief test description.

        Pool window: 2x2
        Pool stride: 1x1
        Pool mode: max
        Padding: none

        Input data:
        FM: 0 BATCH: 0       FM: 1 BATCH: 0
        [-0.5,  0.5,  0.0]   [-1.5, -0.5,  0.0]
        [ 1.0, -1.0, -2.0]   [ 0.0, -1.0,  1.5]
        [-1.0, -0.5, -0.5]   [-2.0,  1.0, -0.5]

        FM: 0 BATCH: 1       FM: 1 BATCH: 1
        [ 0.5,  0.0, -0.5]   [ 0.0,  0.5, -0.5]
        [-2.0, -1.0,  1.0]   [ 1.0, -1.0,  0.0]
        [-0.5, -1.0,  1.5]   [ 0.5, -0.5,  0.0]

        Expected output:
        FM: 0 BATCH: 0       FM: 1 BATCH: 0 
        [ 1.0,  0.5]         [ 0.0,  1.5]   
        [ 1.0, -0.5]         [ 1.0,  1.5]   
                             
        FM: 0 BATCH: 1       FM: 1 BATCH: 1 
        [ 0.5,  1.0]         [ 1.0,  0.5]   
        [-0.5,  1.5]         [ 1.0,  0.0] 
    */

    auto input_prim  = memory::create({engine::cpu, memory::format::yxfb_f32, {3, 3, 2, 2}, true});
    auto output_prim = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 2, 2}, true});
    auto pool_prim = pooling::create({engine::reference, pooling::mode::max, output_prim, input_prim, 1, 2, padding::type::zero});

    auto& input_memory = input_prim.as<const memory&>();
    auto& output_memory = output_prim.as<const memory&>();

    input_memory.set_value<float>( 0, -0.5f); input_memory.set_value<float>( 2, -1.5f);
    input_memory.set_value<float>( 4,  0.5f); input_memory.set_value<float>( 6, -0.5f);
    input_memory.set_value<float>( 8,  0.0f); input_memory.set_value<float>(10,  0.0f);
    input_memory.set_value<float>(12,  1.0f); input_memory.set_value<float>(14,  0.0f);
    input_memory.set_value<float>(16, -1.0f); input_memory.set_value<float>(18, -1.0f);
    input_memory.set_value<float>(20, -2.0f); input_memory.set_value<float>(22,  1.5f);
    input_memory.set_value<float>(24, -1.0f); input_memory.set_value<float>(26, -2.0f);
    input_memory.set_value<float>(28, -0.5f); input_memory.set_value<float>(30,  1.0f);
    input_memory.set_value<float>(32, -0.5f); input_memory.set_value<float>(34, -0.5f);

    input_memory.set_value<float>( 1,  0.5f); input_memory.set_value<float>( 3,  0.0f);
    input_memory.set_value<float>( 5,  0.0f); input_memory.set_value<float>( 7,  0.5f);
    input_memory.set_value<float>( 9, -0.5f); input_memory.set_value<float>(11, -0.5f);
    input_memory.set_value<float>(13, -2.0f); input_memory.set_value<float>(15,  1.0f);
    input_memory.set_value<float>(17, -1.0f); input_memory.set_value<float>(19, -1.0f);
    input_memory.set_value<float>(21,  1.0f); input_memory.set_value<float>(23,  0.0f);
    input_memory.set_value<float>(25, -0.5f); input_memory.set_value<float>(27,  0.5f);
    input_memory.set_value<float>(29, -1.0f); input_memory.set_value<float>(31, -0.5f);
    input_memory.set_value<float>(33,  1.5f); input_memory.set_value<float>(35,  0.0f);

    output_memory.fill<float>(0.0f);

    execute({pool_prim});

    EXPECT_EQ( 1.0f, output_memory.get_value<float>( 0)); EXPECT_EQ( 0.0f, output_memory.get_value<float>( 2));
    EXPECT_EQ( 0.5f, output_memory.get_value<float>( 4)); EXPECT_EQ( 1.5f, output_memory.get_value<float>( 6));
    EXPECT_EQ( 1.0f, output_memory.get_value<float>( 8)); EXPECT_EQ( 1.0f, output_memory.get_value<float>(10));
    EXPECT_EQ(-0.5f, output_memory.get_value<float>(12)); EXPECT_EQ( 1.5f, output_memory.get_value<float>(14));

    EXPECT_EQ( 0.5f, output_memory.get_value<float>( 1)); EXPECT_EQ( 1.0f, output_memory.get_value<float>( 3));
    EXPECT_EQ( 1.0f, output_memory.get_value<float>( 5)); EXPECT_EQ( 0.5f, output_memory.get_value<float>( 7));
    EXPECT_EQ(-0.5f, output_memory.get_value<float>( 9)); EXPECT_EQ( 1.0f, output_memory.get_value<float>(11));
    EXPECT_EQ( 1.5f, output_memory.get_value<float>(13)); EXPECT_EQ( 0.0f, output_memory.get_value<float>(15)); 
}

TEST(pooling_forward, basic_max_yxfb_f32_wsiz4x4_wstr1x1_i2x2x1x1_zeropad1x1) {
    /*  Brief test description.

        Pool window: 4x4
        Pool stride: 1x1
        Pool mode: max
        Padding: zero, 1x1

        Input data:
        [ pad,  pad,  pad, pad]
        [ pad, -0.5,  0.5, pad]
        [ pad,  1.0, -1.0, pad]
        [ pad,  pad,  pad, pad]

        Expected output:
        [ 1.0]
    */

    auto input_prim  = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto output_prim = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto pool_prim = pooling::create({engine::reference, pooling::mode::max, output_prim, input_prim, {-1, -1, 0, 0}, 1, 4, padding::type::zero});

    auto& input_memory = input_prim.as<const memory&>();
    auto& output_memory = output_prim.as<const memory&>();

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  0.5f);
    input_memory.set_value<float>(2,  1.0f);
    input_memory.set_value<float>(3, -1.0f);

    output_memory.fill<float>(0.0f);

    execute({pool_prim});

    EXPECT_EQ(1.0f, output_memory.get_value<float>(0));
}

TEST(pooling_forward, basic_max_yxfb_f32_wsiz3x3_wstr1x1_i2x2x1x1_zeropad1x1) {
    /*  Brief test description.

        Pool window: 3x3
        Pool stride: 1x1
        Pool mode: max
        Padding: zero, 1x1

        Input data:
        [ pad,  pad,  pad, pad]
        [ pad, -0.5,  0.5, pad]
        [ pad,  1.0, -1.0, pad]
        [ pad,  pad,  pad, pad]

        Expected output:
        [ 1.0,  1.0]
        [ 1.0,  1.0]
    */

    auto input_prim  = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto output_prim = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto pool_prim = pooling::create({engine::reference, pooling::mode::max, output_prim, input_prim, {-1, -1, 0, 0}, 1, 3, padding::type::zero});

    auto& input_memory = input_prim.as<const memory&>();
    auto& output_memory = output_prim.as<const memory&>();

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  0.5f);
    input_memory.set_value<float>(2,  1.0f);
    input_memory.set_value<float>(3, -1.0f);

    output_memory.fill<float>(0.0f);

    execute({pool_prim});

    EXPECT_EQ(1.0f, output_memory.get_value<float>(0));
    EXPECT_EQ(1.0f, output_memory.get_value<float>(1));
    EXPECT_EQ(1.0f, output_memory.get_value<float>(2));
    EXPECT_EQ(1.0f, output_memory.get_value<float>(3));
}

TEST(pooling_forward, basic_max_yxfb_f32_wsiz2x2_wstr2x2_i2x2x1x1_zeropad1x1) {
    /*  Brief test description.

        Pool window: 2x2
        Pool stride: 2x2
        Pool mode: max
        Padding: zero, 1x1

        Input data:
        [ pad,  pad,  pad, pad]
        [ pad, -0.5,  0.5, pad]
        [ pad,  1.0, -1.0, pad]
        [ pad,  pad,  pad, pad]

        Expected output:
        [ 0.0,  0.5]
        [ 1.0,  0.0]
    */

    auto input_prim  = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto output_prim = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 1, 1}, true});
    auto pool_prim = pooling::create({engine::reference, pooling::mode::max, output_prim, input_prim, {-1, -1, 0, 0}, 2, 2, padding::type::zero});

    auto& input_memory = input_prim.as<const memory&>();
    auto& output_memory = output_prim.as<const memory&>();

    input_memory.set_value<float>(0, -0.5f);
    input_memory.set_value<float>(1,  0.5f);
    input_memory.set_value<float>(2,  1.0f);
    input_memory.set_value<float>(3, -1.0f);

    output_memory.fill<float>(0.0f);

    execute({pool_prim});

    EXPECT_EQ(0.0f, output_memory.get_value<float>(0));
    EXPECT_EQ(0.5f, output_memory.get_value<float>(1));
    EXPECT_EQ(1.0f, output_memory.get_value<float>(2));
    EXPECT_EQ(0.0f, output_memory.get_value<float>(3));
}

TEST(pooling_forward, basic_max_yxfb_f32_wsiz2x2_wstr2x2_i2x2x2x2_zeropad1x1) {
    /*  Brief test description.

        Pool window: 2x2
        Pool stride: 2x2
        Pool mode: max
        Padding: zero, 1x1

        Input data:
        FM: 0 BATCH: 0           FM: 1 BATCH: 0
        [ pad,  pad,  pad, pad]  [ pad,  pad,  pad, pad]
        [ pad, -0.5,  0.5, pad]  [ pad, -1.5, -0.5, pad]
        [ pad,  1.0, -1.0, pad]  [ pad,  1.0,  1.5, pad]
        [ pad,  pad,  pad, pad]  [ pad,  pad,  pad, pad]

        FM: 0 BATCH: 1           FM: 1 BATCH: 1
        [ pad,  pad,  pad, pad]  [ pad,  pad,  pad, pad]
        [ pad,  0.5, -0.5, pad]  [ pad,  0.5, -0.5, pad]
        [ pad, -1.0,  1.0, pad]  [ pad,  1.0, -1.0, pad]
        [ pad,  pad,  pad, pad]  [ pad,  pad,  pad, pad]

        Expected output:
        FM: 0 BATCH: 0           FM: 1 BATCH: 0 
        [ 0.0,  0.5]             [ 0.0,  0.0]   
        [ 1.0,  0.0]             [ 1.0,  1.5]   

        FM: 0 BATCH: 1           FM: 1 BATCH: 1 
        [ 0.5,  0.0]             [ 0.5,  0.0]   
        [ 0.0,  1.0]             [ 1.0,  0.0] 
    */

    auto input_prim  = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 2, 2}, true});
    auto output_prim = memory::create({engine::cpu, memory::format::yxfb_f32, {2, 2, 2, 2}, true});
    auto pool_prim = pooling::create({engine::reference, pooling::mode::max, output_prim, input_prim, {-1, -1, 0, 0}, 2, 2, padding::type::zero});

    auto& input_memory = input_prim.as<const memory&>();
    auto& output_memory = output_prim.as<const memory&>();

    input_memory.set_value<float>( 0, -0.5f); input_memory.set_value<float>( 2, -1.5f);
    input_memory.set_value<float>( 4,  0.5f); input_memory.set_value<float>( 6, -0.5f);
    input_memory.set_value<float>( 8,  1.0f); input_memory.set_value<float>(10,  1.0f);
    input_memory.set_value<float>(12, -1.0f); input_memory.set_value<float>(14,  1.5f);

    input_memory.set_value<float>( 1,  0.5f); input_memory.set_value<float>( 3,  0.5f);
    input_memory.set_value<float>( 5, -0.5f); input_memory.set_value<float>( 7, -0.5f);
    input_memory.set_value<float>( 9, -1.0f); input_memory.set_value<float>(11,  1.0f);
    input_memory.set_value<float>(13,  1.0f); input_memory.set_value<float>(15, -1.0f);

    output_memory.fill<float>(0.0f);

    execute({pool_prim});

    EXPECT_EQ(0.0f, output_memory.get_value<float>( 0)); EXPECT_EQ(0.0f, output_memory.get_value<float>( 2));
    EXPECT_EQ(0.5f, output_memory.get_value<float>( 4)); EXPECT_EQ(0.0f, output_memory.get_value<float>( 6));
    EXPECT_EQ(1.0f, output_memory.get_value<float>( 8)); EXPECT_EQ(1.0f, output_memory.get_value<float>(10));
    EXPECT_EQ(0.0f, output_memory.get_value<float>(12)); EXPECT_EQ(1.5f, output_memory.get_value<float>(14));

    EXPECT_EQ(0.5f, output_memory.get_value<float>( 1)); EXPECT_EQ(0.5f, output_memory.get_value<float>( 3));
    EXPECT_EQ(0.0f, output_memory.get_value<float>( 5)); EXPECT_EQ(0.0f, output_memory.get_value<float>( 7));
    EXPECT_EQ(0.0f, output_memory.get_value<float>( 9)); EXPECT_EQ(1.0f, output_memory.get_value<float>(11));
    EXPECT_EQ(1.0f, output_memory.get_value<float>(13)); EXPECT_EQ(0.0f, output_memory.get_value<float>(15)); 
}