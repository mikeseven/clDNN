/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "api/neural.h"
#include "gpu/ocl_toolkit.h"
#include <gtest/gtest.h>
#include "test_utils/test_utils.h"
#include "memory_utils.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <thread>

#include "multidimensional_counter.h"//todo remove


using namespace neural;
using namespace tests;

namespace
{
struct gpu_info_helper : gpu::context_holder
{
    gpu::engine_info get_engine_info() const
    {
        return context()->get_engine_info();
    }
};
}

#define USE_RANDOM_SEED 0
#if USE_RANDOM_SEED
std::random_device rnd_device;
unsigned int const random_seed = rnd_device();
#else
unsigned int const random_seed = 1337;
#endif

using VF = std::vector<float>;		// vector
using VVF = std::vector<VF>;		// feature map
using VVVF = std::vector<VVF>;		// 3d feature map
using VVVVF = std::vector<VVVF>;	// batch of 3d feature maps

template<typename T>
T kahan_summation(std::vector<T> &input) {
	std::sort(input.begin(), input.end(), [](T lhs, T rhs) { return std::abs(lhs) < std::abs(rhs); });
	T sum = 0;
	T c = 0;
	for (T x : input) {
		T y = x - c;
		T t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
	return sum;
}

VVF convolve(VVVF &input, VVVF &filter, size_t stride, float bias, size_t f_begin = 0) {
	VVF output;
	size_t output_y = 1 + (input[0].size() - filter[0].size()) / stride;
	size_t output_x = 1 + (input[0][0].size() - filter[0][0].size()) / stride;
	output.assign(output_y, VF(output_x, bias));
	for (size_t f = 0; f < filter.size(); ++f) {
		for (size_t y = 0; y < output_y; ++y) {
			for (size_t x = 0; x < output_x; ++x) {
				VF values;
				values.reserve(filter[0].size() * filter[0][0].size() + 1);
				for (size_t yf = 0; yf < filter[0].size(); ++yf) {
					for (size_t xf = 0; xf < filter[0][0].size(); ++xf) {
						values.push_back(input[f_begin + f][yf + stride * y][xf + stride * x] * filter[f][yf][xf]);
					}
				}
				values.push_back(bias);
				output[y][x] = kahan_summation<float>(values);
			}
		}
	}
	return output;
}

VF flatten_4d(neural::memory::format::type format, VVVVF &data) {
	size_t a = data.size();
	size_t b = data[0].size();
	size_t c = data[0][0].size();
	size_t d = data[0][0][0].size();
	VF vec(a * b * c * d, 0.0f);
	size_t idx = 0;
	if (format == memory::format::yxfb_f32) {
		for (size_t yi = 0; yi < c; ++yi)
			for (size_t xi = 0; xi < d; ++xi)
				for (size_t fi = 0; fi < b; ++fi)
					for (size_t bi = 0; bi < a; ++bi)
						vec[idx++] = data[bi][fi][yi][xi];
	}
	else if (format == memory::format::oiyx_f32) {
		for (size_t oi = 0; oi < a; ++oi)
			for (size_t ii = 0; ii < b; ++ii)
				for (size_t yi = 0; yi < c; ++yi)
					for (size_t xi = 0; xi < d; ++xi)
						vec[idx++] = data[oi][ii][yi][xi];
	}
	return vec;
}

template<typename T>
std::vector<T> generate_random_1d(size_t a, T min, T max) {
	static std::default_random_engine generator(random_seed);
	std::uniform_real_distribution<T> distribution(min, max);
	std::vector<T> v(a);
	for (size_t i = 0; i < a; ++i)
		v[i] = distribution(generator);
	return v;
}

template<typename T>
std::vector<std::vector<T>> generate_random_2d(size_t a, size_t b, T min, T max) {
	std::vector<std::vector<T>> v(a);
	for (size_t i = 0; i < a; ++i)
		v[i] = generate_random_1d(b, min, max);
	return v;
}

template<typename T>
std::vector<std::vector<std::vector<T>>> generate_random_3d(size_t a, size_t b, size_t c, T min, T max) {
	std::vector<std::vector<std::vector<T>>> v(a);
	for (size_t i = 0; i < a; ++i)
		v[i] = generate_random_2d(b, c, min, max);
	return v;
}

// parameters order is assumed to be bfyx or oiyx
template<typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> generate_random_4d(size_t a, size_t b, size_t c, size_t d, T min, T max) {
	std::vector<std::vector<std::vector<std::vector<T>>>> v(a);
	for (size_t i = 0; i < a; ++i)
		v[i] = generate_random_3d(b, c, d, min, max);
	return v;
}

// rounds floating point number, fraction precision should be in the range [0,23]
// masks the bits:
// 1 11111111 11111111111111100000000
// |      |            |
// sign  exp        fraction
float float_round(float x, size_t fraction_precision = 15) {
	uint32_t mask = ~((1 << (23 - fraction_precision)) - 1);
	reinterpret_cast<uint32_t&>(x) &= mask;
	return x;
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad_random) {
	//  Filter : 2x2
	//  Stride : 2x2
	//  Input  : 4x4
	//  Output : 2x2
	//
	//  Input:
	//  rnd  rnd  rnd  rnd
	//  rnd  rnd  rnd  rnd
	//  rnd  rnd  rnd  rnd
	//  rnd  rnd  rnd  rnd
	//
	//  Filter
	//  rnd  rnd
	//  rnd  rnd
	//
	//  Bias
	//  rnd
	//
	//  Output:
	//  rnd  rnd
	//  rnd  rnd

	size_t batch = 1, input_f = 1, input_y = 4, input_x = 4;

	VVVVF input_rnd = generate_random_4d<float>(batch, input_f, input_y, input_x, -10.0f, 10.0f);
	VF input_rnd_vec = flatten_4d(memory::format::yxfb_f32, input_rnd);
	VVVVF filter_rnd = generate_random_4d<float>(1, 1, 2, 2, -10.0f, 10.0f);
	VF filter_rnd_vec = flatten_4d(memory::format::oiyx_f32, filter_rnd);
	VF bias_rnd = generate_random_1d<float>(1, -10.0f, 10.0f);
	VVVVF output_rnd(batch, VVVF(filter_rnd.size()));
	for (size_t b = 0; b < output_rnd.size(); ++b) {
		for (size_t of = 0; of < filter_rnd.size(); ++of) {
			output_rnd[b][of] = convolve(input_rnd[b], filter_rnd[of], 2, bias_rnd[of]);
		}
	}
	VF output_rnd_vec = flatten_4d(memory::format::yxfb_f32, output_rnd);

	auto input = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 4, 4 }, 1 } });
	auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
	auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
	auto biases = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } } , 1 } });
	set_values(input, input_rnd_vec);
	set_values(weights, filter_rnd_vec);
	set_values(biases, bias_rnd);
	auto conv = convolution::create({ output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });
	execute({ conv }).wait();
	auto output_ptr = output.as<const memory&>().pointer<float>();

	for (size_t i = 0; i < output_rnd.size(); ++i) {
		float x = float_round(output_rnd_vec[i]), y = float_round(output_ptr[i]);
		EXPECT_FLOAT_EQ(x, y) << "random seed = " << random_seed << std::endl;
	}
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad_random) {
	//  Filter : 2x2
	//  Stride : 2x2
	//  Input  : 2x2x1x2
	//  Output : 1x1x1x2
	//
	//  Input:
	//  rnd  rnd    rnd  rnd
	//  rnd  rnd    rnd  rnd
	//
	//  Filter:
	//  rnd  rnd
	//  rnd  rnd
	//
	//  Bias:
	//  rnd
	//
	//  Output:
	//  rnd  rnd

	size_t batch = 2, input_f = 1, input_y = 2, input_x = 2;

	VVVVF input_rnd = generate_random_4d<float>(batch, input_f, input_y, input_x, -10.0f, 10.0f);
	VF input_rnd_vec = flatten_4d(memory::format::yxfb_f32, input_rnd);
	VVVVF filter_rnd = generate_random_4d<float>(1, 1, 2, 2, -10.0f, 10.0f);
	VF filter_rnd_vec = flatten_4d(memory::format::oiyx_f32, filter_rnd);
	VF bias_rnd = generate_random_1d<float>(1, -10.0f, 10.0f);
	VVVVF output_rnd(batch, VVVF(filter_rnd.size()));
	for (size_t b = 0; b < output_rnd.size(); ++b) {
		for (size_t of = 0; of < filter_rnd.size(); ++of) {
			output_rnd[b][of] = convolve(input_rnd[b], filter_rnd[of], 2, bias_rnd[of]);
		}
	}
	VF output_rnd_vec = flatten_4d(memory::format::yxfb_f32, output_rnd);

	auto input = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 2, 2 }, 1 } });
	auto output = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 1, 1 }, 1 } });
	auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
	auto biases = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } } , 1 } });
	set_values(input, input_rnd_vec);
	set_values(weights, filter_rnd_vec);
	set_values(biases, bias_rnd);
	auto conv = convolution::create({ output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });
	execute({ conv }).wait();
	auto output_ptr = output.as<const memory&>().pointer<float>();

	for (size_t i = 0; i < output_rnd.size(); ++i) {
		float x = float_round(output_rnd_vec[i]), y = float_round(output_ptr[i]);
		EXPECT_FLOAT_EQ(x, y) << "random seed = " << random_seed << std::endl;
	}
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //
    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  8  0.5
    //  6  9

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 4, 4 }, 1 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    auto conv = convolution::create({ output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(8.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.5f, output_ptr[1]);
    EXPECT_FLOAT_EQ(6.0f, output_ptr[2]);
    EXPECT_FLOAT_EQ(9.0f, output_ptr[3]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x2x1x2
    //  Output : 1x1x1x2
    //
    //  Input:
    //  0.5   1.5    2.3 -0.4
    //  2.0  -4.0    1.0  3.0
    //
    //  Filter:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias:
    //  -1
    //
    //  Output:
    //  3.65 -5.36
    auto input = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 2, 2 }, 1 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 1, 1 }, 1 } });
    auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, { 0.5f, 2.3f, 1.5f, -0.4f, 2.0f, 1.0f, -4.0f, 3.0f });
    set_values(weights, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases, { -1.0f });

    auto conv = convolution::create({ output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(3.65f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.36f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz2x1x2x1_in1x2x1_nopad) {
    //  Filter : 1x2x1x2x1
    //  Input  : 1x1x2x1
    //  Output : 1x2x1x1
    //
    //  Input:
    //  1.0    2.0
    //
    // Filter:
    //   1.0    2.0  ofm=0
    //  -1.0   -2.0  ofm=1
    //
    //  Bias:
    //  0.1 -0.2
    //
    //  Output:
    //   5.1  f=0
    //  -5.2  f=1

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 1 }, 1 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 2 } });
    auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1 ,{ 2, 1 },{ 2, 1 } } });
    auto biases = memory::allocate({ memory::format::x_f32,{ 1 ,{ { 2 } }, 1 } });

    set_values(input, { 1.0f, 2.0f });
    set_values(weights, { 1.0f, 2.0f, -1.0f, -2.0f });
    set_values(biases, { 0.1f, -0.2f });

    auto conv = convolution::create({ output,{ input, weights, biases },{ 1,{ 5, 5 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(5.1f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.2f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz3x2x2x1_in2x2x1_nopad) {
    //  Filter : 1x3x2x2x1
    //  Input  : 1x2x2x1
    //  Output : 1x3x1x1
    //
    //  Input:
    //  1.0    2.0  f=0
    //  3.0    4.0  f=1
    //
    // Filter:
    //   1.0    2.0  ifm=0  ofm=0
    //   3.0    4.0  ifm=1
    //
    //   5.0    6.0  ifm=0  ofm=1
    //   7.0    8.0  ifm=1
    //
    //   9.0   10.0  ifm=0  ofm=2
    //  11.0   12.0  ifm=1
    //  Bias:
    //   -5     -6     -7
    //
    //  Output:
    //   25.0  f=0
    //   64,0  f=1
    //  103.0  f=2

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 1 }, 2 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 3 } });
    auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1 ,{ 2, 1 },{ 3, 2 } } });
    auto biases = memory::allocate({ memory::format::x_f32,{ 1 ,{ { 3 } }, 1 } });

    set_values(input, { 1.0f, 3.0f, 2.0f, 4.0f });
    set_values(weights, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });
    set_values(biases, { -5.0f, -6.0f, -7.0f });

    auto conv = convolution::create({ output,{ input, weights, biases },{ 1,{ 5, 5 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(25.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(64.0f, output_ptr[1]);
    EXPECT_FLOAT_EQ(103.0f, output_ptr[2]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2x1x3_wstr2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2x1x3
    //  Stride : 2x2
    //  Input  : 2x2x1x1
    //  Output : 1x1x3x1
    //
    //  Input:
    //  -2.3 -0.1
    //   3.1  1.9
    //
    //  Filter:
    //  -1.1  1.5       0.1  0.2        2.0  -1.0
    //   0.5 -0.5       0.4  0.7        2.5  -1.5
    //
    //  Bias:
    //  0.1 -0.2 0.3
    //
    //  Output:
    //   0.7
    //   2.12
    //   3.08

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 3 } });
    auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1 ,{ 2, 2 },{ 3, 1 } } });
    auto biases = memory::allocate({ memory::format::x_f32,{ 1 ,{ { 3 } }, 1 } });

    set_values(input, { -2.3f, -0.1f, 3.1f, 1.9f });
    set_values(weights, { -1.1f, 1.5f, 0.5f, -0.5f, 0.1f, 0.2f, 0.4f, 0.7f, 2.0f, -1.0f, 2.5f, -1.5f });
    set_values(biases, { 0.1f, -0.2f, 0.3f });

    auto conv = convolution::create({ output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_TRUE(are_equal(3.08f, output_ptr[0]));
    EXPECT_TRUE(are_equal(2.12f, output_ptr[1]));
    EXPECT_TRUE(are_equal(0.7f,  output_ptr[2]));
}

TEST(convolution_f32_fw_gpu, wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
    //  Filter  : 3x3
    //  Stride  : 2x2
    //  Input   : 2x2
    //  Output  : 1x1
    //  Padding : zero
    //
    //  Input:
    //  -0.5   1.0   padd
    //   0.5   2.0   padd
    //  padd  padd   padd
    //
    //  Filter
    //  -2    0.5  3.5
    //   1.5  4   -5
    //   0.5  1.5 -1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  12.25
    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 1 } });
    auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 3, 3 },{ 1, 1 } } });
    auto biases = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    auto conv = convolution::create({ output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero });
    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(12.25f, output_ptr[0]);
}

TEST(convolution_f32_fw_gpu, offsets_wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
    //   Filter       : 3x3
    //   Stride       : 2x2
    //   Input        : 2x2
    //   Input offset : -1x-1
    //   Output       : 2x2
    //   Output offset: 1x1
    //   Padding      : zero
    //
    //   Input:
    //   padd padd  padd
    //   padd -0.5   1
    //   padd  0.5   2.0
    //
    //   Filter
    //   -2    0.5  3.5
    //    1.5  4   -5
    //    0.5  1.5 -1.5
    //
    //   Bias
    //   2
    //
    //   Output:
    //   rnd   rnd
    //   rnd   2.0
    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1 ,{ 3, 3 },{ 1, 1 } } });
    auto biases = memory::allocate({ memory::format::x_f32,{ 1 ,{ { 1 } } , 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    auto conv = convolution::create({ 
        output,
        { 0,{ 1, 1 }, 0 },
        { 1,{ 1, 1 }, 1 },
        { input, weights, biases },
        { 0,{ -1, -1 }, 0 },
        { 1,{ 2,  2 }, 1 },
        padding::zero });
    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(2.0f, output_ptr[3]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_split2) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4x2
    //  Output : 2x2x2
    //
    //  Input:
    //  f0: -0.5   1     0.5  2
    //       1.5  -0.5   0   -1
    //       0.5   0.5  -1    1
    //       0.5   2     1.5 -0.5
    //
    //  f1:  0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //       0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //
    //  Filter1:
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias1:
    //  2
    //
    //  Filter2:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias2:
    //  -1

    //  Output:
    //   8  3.65 0.5 -5.36
    //   6  3.65 9   -5.36

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 4, 4 }, 2 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 2 } });
    auto weights1 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases1 = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } } , 1 } });
    auto weights2 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases2 = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, {
        -0.5f,  0.5f,  1.0f,  1.5f,  0.5f,  2.3f,  2.0f, -0.4f,
        1.5f,  2.0f, -0.5f, -4.0f,  0.0f,  1.0f, -1.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  1.5f, -1.0f,  2.3f,  1.0f, -0.4f,
        0.5f,  2.0f,  2.0f, -4.0f,  1.5f,  1.0f, -0.5f,  3.0f
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases1, { 2.0f });
    set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases2, { -1.0f });

    auto conv = convolution::create({
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2) {
    //  2x Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x4x4x2
    //  Output : 2x2x2x2
    //
    //  Input:
    //  f0b0: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f0b1: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f1b0:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
    //  f1b1:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
    //
    //  Filter1:
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias1:
    //  2
    //
    //  Filter2:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias2:
    //  -1

    //  Output:
    //   8  8 3.65 3.65 0.5  0.5 -5.36 -5.36
    //   6  6 3.65 3.65 9    9   -5.36 -5.36

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 4, 4 }, 2 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 } });
    auto weights1 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases1 = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } } , 1 } });
    auto weights2 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases2 = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } } , 1 } });

    set_values(input, {
       -0.5f, -0.5f,  0.5f,  0.5f,  1.0f,  1.0f,  1.5f,  1.5f,  0.5f,  0.5f,  2.3f,  2.3f,  2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, -0.5f, -0.5f, -4.0f, -4.0f,  0.0f,  0.0f,  1.0f,  1.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  1.5f,  1.5f, -1.0f, -1.0f,  2.3f,  2.3f,  1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f,  2.0f,  2.0f, -4.0f, -4.0f,  1.5f,  1.5f,  1.0f,  1.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases1, { 2.0f });
    set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases2, { -1.0f });

    auto conv = convolution::create({
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 8));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 9));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 10));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 11));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 12));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 13));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 14));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 15));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x4x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x4
    //  Output : 1x1x4
    //
    //  Input:
    //  f0:  1.5
    //  f1:  0.5
    //
    //  f2:  0.0
    //  f3: -0.5
    //
    //
    //  Filter1:
    //  -2 -0.5  ofm=0
    //   1  2    ofm=1 
    //  Bias1:
    //   1  5
    //
    //  Filter2:
    //   4  1.5  ofm=0
    //   2  0.5  ofm=1
    //
    //  Bias2:
    //  -1  2.5
    //
    //  Output:
    //  -2.25  
    //   7.5
    //
    //  -1.75
    //   2.25
    

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto weights1 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 2, 2 } } });
    auto biases1 = memory::allocate({ memory::format::x_f32,{ 1,{ { 2 } } , 1 } });
    auto weights2 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 2, 2 } } });
    auto biases2 = memory::allocate({ memory::format::x_f32,{ 1,{ { 2 } } , 1 } });

    set_values(input, {
       1.5f, 0.5f, 0.0f, -0.5f
    });
    set_values(weights1, { -2.0f, -0.5f, 1.0f, 2.0f });
    set_values(biases1, { 1.0f, 5.0f });
    set_values(weights2, { 4.0f, 1.5f, 2.0f, 0.5f });
    set_values(biases2, { -1.0f, 2.5f });

    auto conv = convolution::create({
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(-2.25f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(7.5f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(-1.75f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(2.25f, get_value<float>(output_ptr, 3));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x2x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x2
    //  Output : 1x1x4
    //
    //  Input:
    //  f0:  1.5
    //
    //  f1:  0.5
    //
    //  Filter1:
    //  -2  ofm=0
    //   1  ofm=1 
    //  Bias1:
    //   1  5
    //
    //  Filter2:
    //   4  ofm=0
    //   2  ofm=1
    //
    //  Bias2:
    //  -1  2.5
    //
    //  Output:
    //  -2  
    //   6.5
    //
    //   1
    //   3.5


    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 2 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto weights1 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 2, 1 } } });
    auto biases1 = memory::allocate({ memory::format::x_f32,{ 1,{ { 2 } } , 1 } });
    auto weights2 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 2, 1 } } });
    auto biases2 = memory::allocate({ memory::format::x_f32,{ 1,{ { 2 } } , 1 } });

    set_values(input, {
        1.5f, 0.5f
    });
    set_values(weights1, { -2.0f, 1.0f });
    set_values(biases1, { 1.0f, 5.0f });
    set_values(weights2, { 4.0f, 2.0f });
    set_values(biases2, { -1.0f, 2.5f });

    auto conv = convolution::create({
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(-2.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(6.5f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(1.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(3.5f, get_value<float>(output_ptr, 3));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x4x1_filter_1x3x2x1x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x4
    //  Output : 1x1x6
    //
    //  Input:
    //  f0:  1.5
    //  f1:  0.5
    //
    //  f2:  2
    //  f3: -1.0
    //
    //  Filter1:
    //  -2   1   ofm=0
    //   1   3   ofm=1
    //   0.5 8   ofm=2
    //  Bias1:
    //   1   5   3
    //
    //  Filter2:
    //   4  -4   ofm=0
    //   2   0.5 ofm=1
    //  -0.5 3   ofm=2
    //
    //  Bias2:
    //  -1   2.5 2
    //
    //  Output:
    //  -1.5  
    //   8
    //   7.75
    //
    //   11
    //   6
    //  -2


    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 6 } });
    auto weights1 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 3, 2 } } });
    auto biases1 = memory::allocate({ memory::format::x_f32,{ 1,{ { 3 } } , 1 } });
    auto weights2 = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 1, 1 },{ 3, 2 } } });
    auto biases2 = memory::allocate({ memory::format::x_f32,{ 1,{ { 3 } } , 1 } });

    set_values(input, {
        1.5f, 0.5f, 2.0f, -1.0f
    });
    set_values(weights1, { -2.0f, 1.0f, 1.0f, 3.0f, 0.5f, 8.0f });
    set_values(biases1, { 1.0f, 5.0f, 3.0f });
    set_values(weights2, { 4.0f, -4.0f, 2.0f, 0.5f, -0.5f, 3.0f });
    set_values(biases2, { -1.0f, 2.5f, 2.0f });

    auto conv = convolution::create({
        output,
        { input, weights1, biases1, weights2, biases2 },
        { 1,{ 2, 2 }, 1 },
        padding::zero,
        2
    });

    execute({ conv }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(-1.5f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(7.75f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(11.0f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(-2.0f, get_value<float>(output_ptr, 5));

}

TEST(convolution_gpu, trivial_convolution_relu) {

    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2

    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  -2
    //
    //  Output:
    //  4  0.0
    //  2  5

    auto input = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 4, 4 }, 1 } });
    auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
    auto biases = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } }, 1 } });

    set_values(input, {
        -0.5f,  1.0f,  0.5f,  2.0f,
        1.5f, -0.5f,  0.0f, -1.0f,
        0.5f,  0.5f, -1.0f,  1.0f,
        0.5f,  2.0f,  1.5f, -0.5f
    });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { -2.0f });

    auto conv_relu = convolution::create({ output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero, 1, true, 0 });
    execute({ conv_relu }).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();
    EXPECT_FLOAT_EQ(4.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(0.0f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(2.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(5.0f, get_value<float>(output_ptr, 3));
}

TEST(convolution_gpu, relu_with_negative_slope) {

	//  Filter : 2x2
	//  Stride : 2x2
	//  Input  : 4x4
	//  Output : 2x2
	//  Negative Slope : 0.1

	//  Input:
	//  -0.5   1     0.5  2
	//   1.5  -0.5   0   -1
	//   0.5   0.5  -1    1
	//   0.5   2     1.5 -0.5
	//
	//  Filter
	//  -2   0.5
	//   3.5 1.5
	//
	//  Bias
	//  -2
	//
	//  Output:
	//  4  -0.35
	//  2  5

	auto input = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 4, 4 }, 1 } });
	auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
	auto weights = memory::allocate({ memory::format::oiyx_f32,{ 1,{ 2, 2 },{ 1, 1 } } });
	auto biases = memory::allocate({ memory::format::x_f32,{ 1,{ { 1 } }, 1 } });

	set_values(input, {
		-0.5f,  1.0f,  0.5f,  2.0f,
		1.5f, -0.5f,  0.0f, -1.0f,
		0.5f,  0.5f, -1.0f,  1.0f,
		0.5f,  2.0f,  1.5f, -0.5f
	});
	set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
	set_values(biases, { -2.0f });

	auto conv_relu = convolution::create({ output,{ input, weights, biases },{ 1,{ 2, 2 }, 1 }, padding::zero, 1, true, 0.1f });
	execute({ conv_relu }).wait();

	auto output_ptr = output.as<const memory&>().pointer<float>();
	EXPECT_FLOAT_EQ(4.0f, get_value<float>(output_ptr, 0));
	EXPECT_FLOAT_EQ(-0.35f, get_value<float>(output_ptr, 1));
	EXPECT_FLOAT_EQ(2.0f, get_value<float>(output_ptr, 2));
	EXPECT_FLOAT_EQ(5.0f, get_value<float>(output_ptr, 3));
}

TEST(convolution_gpu, basic_yxfb_4_4_yxio_2_2_b16_if2_of16_st2_2_p0_sp1_fp32)
{
#define USE_OLD_WEIGHTS_FORMAT 0

    const auto input_format   = memory::format::yxfb_f32;
#if USE_OLD_WEIGHTS_FORMAT
    const auto weights_format = memory::format::oiyx_f32;
#else
    const auto weights_format = memory::format::yxio_f32;
#endif
    const auto biases_format = memory::format::x_f32;
    const auto output_format  = input_format;

    const uint32_t batch_size = 16;
    const uint32_t input_feature_count = 2;
    const uint32_t output_feature_count = 16;

    const uint32_t stride_x = 2;
    const uint32_t stride_y = 2;

    const uint32_t input_x = 4;
    const uint32_t input_y = 4;
    const uint32_t weights_x = 2;
    const uint32_t weights_y = 2;
    const uint32_t output_x = (input_x - weights_x) / stride_x + 1;
    const uint32_t output_y = (input_y - weights_y) / stride_y + 1;


    auto input = memory::allocate({input_format, {batch_size, {input_x, input_y}, input_feature_count}});
    auto weights = memory::allocate({weights_format, {1, {weights_x, weights_y}, {output_feature_count, input_feature_count}}});
    auto biases = memory::allocate({biases_format, {1, {{output_feature_count}}, 1}});
    auto output = memory::allocate({output_format, {batch_size, {output_x, output_y}, output_feature_count}});


    // input:
    std::vector<float> input_vals_template {
        0.25f, 0.50f, 0.75f, 1.00f,
        1.25f, 1.50f, 1.75f, 2.00f,
        2.25f, 2.50f, 2.75f, 3.00f,
        3.25f, 3.50f, 3.75f, 4.00f,
    };
    input_vals_template.resize(input_y * input_x);

    std::vector<float> input_vals;
    input_vals.reserve(input_y * input_x * input_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < input_y * input_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                input_vals.push_back((bi * input_feature_count + ifi + 1) * input_vals_template[yxi]);
            }
        }
    }
    set_values(input, input_vals);


    // weights:
    std::vector<float> weights_vals_template {
        -4.0f, -2.0f,
         4.0f,  4.0f,
    };
    weights_vals_template.resize(weights_y * weights_x);

    std::vector<float> weights_vals;
    weights_vals.reserve(weights_y * weights_x * input_feature_count * output_feature_count);
#if USE_OLD_WEIGHTS_FORMAT
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#else
    for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#endif
    set_values(weights, weights_vals);


    // biases:
    std::vector<float> biases_vals;
    biases_vals.reserve(output_feature_count);
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        biases_vals.push_back(ofi * 1.0f);
    }
    set_values(biases, biases_vals);


    // output:
    std::vector<float> output_vals_template {
         9.0f, 10.0f,
        13.0f, 14.0f,
    };
    output_vals_template.resize(output_y * output_x);

    std::vector<float> output_vals;
    output_vals.reserve(output_y * output_x * output_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                uint32_t template_factor = input_feature_count * input_feature_count * input_feature_count * bi * ofi +
                    input_feature_count * input_feature_count * (input_feature_count + 1) / 2 * (bi + ofi) +
                    input_feature_count * (input_feature_count + 1) * (2 * input_feature_count + 1) / 6;
                float bias_factor = ofi * 1.0f;

                output_vals.push_back(template_factor * output_vals_template[yxi] + bias_factor);
            }
        }
    }


    // Computing convolution.
    auto conv = convolution::create({output, {input, weights, biases}, {1, {stride_x, stride_y}, 1}, padding::zero});
    execute({conv}).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();


    // Checking result.
    uint32_t i = 0;
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi, ++i)
            {
                auto equal = are_equal(output_vals[i], get_value<float>(output_ptr, i));
                EXPECT_TRUE(equal);
                if (!equal)
                {
                    std::cout << "Failed at position (" << yxi << ", output feature = " << ofi << ", batch = " << bi << "): "
                        << output_vals[i] << " != " << get_value<float>(output_ptr, i) << std::endl;
                    return;
                }
            }
        }
    }

#undef USE_OLD_WEIGHTS_FORMAT
}

TEST(convolution_gpu, basic_yxfb_4_4_yxio_2_2_b16_if2_of16_st2_2_p0_sp1_fp16)
{
#define USE_OLD_WEIGHTS_FORMAT 0

    gpu_info_helper gpu_info;
    if (!gpu_info.get_engine_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }


    const auto input_format   = memory::format::yxfb_f32;
#if USE_OLD_WEIGHTS_FORMAT
    const auto weights_format = memory::format::oiyx_f32;
#else
    const auto weights_format = memory::format::yxio_f32;
#endif
    const auto biases_format  = memory::format::x_f32;
    const auto output_format  = input_format;

    const auto input_cvt_format   = memory::format::yxfb_f16;
#if USE_OLD_WEIGHTS_FORMAT
    const auto weights_cvt_format = memory::format::oiyx_f16;
#else
    const auto weights_cvt_format = memory::format::yxio_f16;
#endif
    const auto biases_cvt_format  = memory::format::x_f16;
    const auto output_cvt_format  = input_cvt_format;

    const uint32_t batch_size = 16;
    const uint32_t input_feature_count = 2;
    const uint32_t output_feature_count = 16;

    const uint32_t stride_x = 2;
    const uint32_t stride_y = 2;

    const uint32_t input_x = 4;
    const uint32_t input_y = 4;
    const uint32_t weights_x = 2;
    const uint32_t weights_y = 2;
    const uint32_t output_x = (input_x - weights_x) / stride_x + 1;
    const uint32_t output_y = (input_y - weights_y) / stride_y + 1;


    auto input   = memory::allocate({input_format, {batch_size, {input_x, input_y}, input_feature_count}});
    auto weights = memory::allocate({weights_format, {1, {weights_x, weights_y}, {output_feature_count, input_feature_count}}});
    auto biases  = memory::allocate({biases_format, {1, {{output_feature_count}}, 1}});
    auto output  = memory::allocate({output_format, {batch_size, {output_x, output_y}, output_feature_count}});

    auto input_cvtd   = memory::allocate({input_cvt_format, {batch_size, {input_x, input_y}, input_feature_count}});
    auto weights_cvtd = memory::allocate({weights_cvt_format, {1, {weights_x, weights_y}, {output_feature_count, input_feature_count}}});
    auto biases_cvtd  = memory::allocate({biases_cvt_format, {1, {{output_feature_count}}, 1}});
    auto output_cvtd  = memory::allocate({output_cvt_format, {batch_size, {output_x, output_y}, output_feature_count}});


    // input:
    std::vector<float> input_vals_template {
        0.25f, 0.50f, 0.75f, 1.00f,
        1.25f, 1.50f, 1.75f, 2.00f,
        2.25f, 2.50f, 2.75f, 3.00f,
        3.25f, 3.50f, 3.75f, 4.00f,
    };
    input_vals_template.resize(input_y * input_x);

    std::vector<float> input_vals;
    input_vals.reserve(input_y * input_x * input_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < input_y * input_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                input_vals.push_back((bi * input_feature_count + ifi + 1) * input_vals_template[yxi]);
            }
        }
    }
    set_values(input, input_vals);


    // weights:
    std::vector<float> weights_vals_template {
        -0.50f, -0.25f,
         0.50f,  0.50f,
    };
    weights_vals_template.resize(weights_y * weights_x);

    std::vector<float> weights_vals;
    weights_vals.reserve(weights_y * weights_x * input_feature_count * output_feature_count);
#if USE_OLD_WEIGHTS_FORMAT
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#else
    for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#endif
    set_values(weights, weights_vals);


    // biases:
    std::vector<float> biases_vals;
    biases_vals.reserve(output_feature_count);
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        biases_vals.push_back(ofi * 1.0f);
    }
    set_values(biases, biases_vals);


    // output:
    std::vector<float> output_vals_template {
        1.125f,  1.250f,
        1.625f,  1.750f,
    };
    output_vals_template.resize(output_y * output_x);

    std::vector<float> output_vals;
    output_vals.reserve(output_y * output_x * output_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                uint32_t template_factor = input_feature_count * input_feature_count * input_feature_count * bi * ofi +
                    input_feature_count * input_feature_count * (input_feature_count + 1) / 2 * (bi + ofi) +
                    input_feature_count * (input_feature_count + 1) * (2 * input_feature_count + 1) / 6;
                float bias_factor = ofi * 1.0f;

                output_vals.push_back(template_factor * output_vals_template[yxi] + bias_factor);
            }
        }
    }

    auto expected_float = memory::allocate({memory::format::x_f32, {1, {{static_cast<uint32_t>(output_vals.size())}}, 1}});
    auto expected_half  = memory::allocate({memory::format::x_f16, {1, {{static_cast<uint32_t>(output_vals.size())}}, 1}});
    auto expected       = memory::allocate({memory::format::x_f32, {1, {{static_cast<uint32_t>(output_vals.size())}}, 1}});

//    set_values(expected_float, output_vals);
//    auto cvt_expected_f32_f16 = reorder::create({expected_float, expected_half});
//    auto cvt_expected_f16_f32 = reorder::create({expected_half, expected});
//    execute({cvt_expected_f32_f16, cvt_expected_f16_f32}).wait();
//
//    auto expected_ptr = expected.as<const memory&>().pointer<float>();


    // Computing convolution.
    auto cvt_input   = reorder::create({input, input_cvtd});
    auto cvt_weights = reorder::create({weights, weights_cvtd});
    auto cvt_biases  = reorder::create({biases, biases_cvtd});
    auto conv        = convolution::create({output_cvtd, {input_cvtd, weights_cvtd, biases_cvtd}, {1, {stride_x, stride_y}, 1}, padding::zero});
    auto cvt_output  = reorder::create({output_cvtd, output});
    execute({cvt_input, cvt_weights, cvt_biases, conv, cvt_output}).wait();

    auto output_ptr = output.as<const memory&>().pointer<float>();

#if 0
    for (uint32_t bi = 0; bi < /*batch_size*/ 4; ++bi)
    {
        std::cout << "Actual (batch: " << std::setw(3) << std::right << bi << "):" << std::endl;
        for (uint32_t yi = 0; yi < output_y; ++yi)
        {
            const char* x_sep = "";
            const char* f_sep = "";
            for (uint32_t ofi = 0; ofi < /*output_feature_count*/ 4; ++ofi)
            {
                std::cout << f_sep;
                f_sep = "  |";
                for (uint32_t xi = 0; xi < output_x; ++xi)
                {
                    uint32_t pos = bi + batch_size * (ofi + output_feature_count * (xi + output_x * yi));
                    auto val = get_value(output_ptr, pos);

                    std::cout << x_sep << std::setw(8) << std::setprecision(2) << std::right << std::fixed << val;
                    x_sep = "  ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Expected (batch: " << std::setw(3) << std::right << bi << "):" << std::endl;
        for (uint32_t yi = 0; yi < output_y; ++yi)
        {
            const char* x_sep = "";
            const char* f_sep = "";
            for (uint32_t ofi = 0; ofi < /*output_feature_count*/ 4; ++ofi)
            {
                std::cout << f_sep;
                f_sep = "  |";
                for (uint32_t xi = 0; xi < output_x; ++xi)
                {
                    uint32_t pos = bi + batch_size * (ofi + output_feature_count * (xi + output_x * yi));
                    auto val = output_vals[pos];
                    //auto val = get_value(expected_ptr, pos);

                    std::cout << x_sep << std::setw(8) << std::setprecision(2) << std::right << std::fixed << val;
                    x_sep = "  ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
#endif 

    // Checking result.
    uint32_t i = 0;
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi, ++i)
            {
                auto equal = are_equal(output_vals[i] /*get_value(expected_ptr, i)*/, get_value(output_ptr, i), 0.002f);
                EXPECT_TRUE(equal);
                if (!equal)
                {
                    std::cout << "Failed at position (" << yxi << ", output feature = " << ofi << ", batch = " << bi << "): "
                        << output_vals[i] /*get_value(expected_ptr, i)*/ << " != " << get_value(output_ptr, i) << std::endl;
                    return;
                }
            }
        }
    }

#undef USE_OLD_WEIGHTS_FORMAT
}
