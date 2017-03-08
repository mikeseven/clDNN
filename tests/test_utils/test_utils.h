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

//todo move to another folder

#pragma once

#include "api/memory.hpp"
#include "api/tensor.hpp"
#include <iostream>
#include <limits>
#include <random>
#include <gtest/gtest.h>
#include <api/primitive.hpp>
#include "float16.h"

namespace tests {

#define USE_RANDOM_SEED 0
#if USE_RANDOM_SEED
	std::random_device rnd_device;
	unsigned int const random_seed = rnd_device();
#else
	unsigned int const random_seed = 1337;
#endif

template<typename T>
using VF = std::vector<T>;		// float vector
template<typename T>
using VVF = std::vector<VF<T>>;		// feature map
template<typename T>
using VVVF = std::vector<VVF<T>>;		// 3d feature map
template<typename T>
using VVVVF = std::vector<VVVF<T>>;	// batch of 3d feature maps
template<typename T>
using VVVVVF = std::vector<VVVVF<T>>;	// split of oiyx filters

template<typename T>
inline VF<T> flatten_4d(cldnn::format input_format, VVVVF<T> &data) {
	size_t a = data.size();
	size_t b = data[0].size();
	size_t c = data[0][0].size();
	size_t d = data[0][0][0].size();
	VF<T> vec(a * b * c * d, 0.0f);
	size_t idx = 0;

	switch (input_format.value) {
		case cldnn::format::yxfb:
			for (size_t yi = 0; yi < c; ++yi)
				for (size_t xi = 0; xi < d; ++xi)
					for (size_t fi = 0; fi < b; ++fi)
						for (size_t bi = 0; bi < a; ++bi)
							vec[idx++] = data[bi][fi][yi][xi];
			break;

		case cldnn::format::oiyx:
    		for (size_t oi = 0; oi < a; ++oi)
    			for (size_t ii = 0; ii < b; ++ii)
    				for (size_t yi = 0; yi < c; ++yi)
    					for (size_t xi = 0; xi < d; ++xi)
    						vec[idx++] = data[oi][ii][yi][xi];
            break;
		
		case cldnn::format::bfyx:
			for (size_t bi = 0; bi < a; ++bi)
				for (size_t fi = 0; fi < b; ++fi)
					for (size_t yi = 0; yi < c; ++yi)
						for (size_t xi = 0; xi < d; ++xi)
							vec[idx++] = data[bi][fi][yi][xi];
			break;
		
		case cldnn::format::yxio:
			for (size_t yi = 0; yi < c; ++yi)
				for (size_t xi = 0; xi < d; ++xi)
					for (size_t ii = 0; ii < b; ++ii)
						for (size_t oi = 0; oi < a; ++oi)															
							vec[idx++] = data[oi][ii][yi][xi];
			break;

		default:
			assert(0);
	}
	return vec;
}

template<typename T>
std::vector<T> generate_random_1d(size_t a, int min, int max) {
	static std::default_random_engine generator(random_seed);
	int k = 8; // 1/k is the resolution of the floating point numbers
	std::uniform_int_distribution<int> distribution(k * min, k * max);
	std::vector<T> v(a);
	for (size_t i = 0; i < a; ++i) {
		v[i] = (T)distribution(generator);
		v[i] /= k;
	}
	return v;
}

template<typename T>
std::vector<std::vector<T>> generate_random_2d(size_t a, size_t b, int min, int max) {
	std::vector<std::vector<T>> v(a);
	for (size_t i = 0; i < a; ++i)
		v[i] = generate_random_1d<T>(b, min, max);
	return v;
}

template<typename T>
std::vector<std::vector<std::vector<T>>> generate_random_3d(size_t a, size_t b, size_t c, int min, int max) {
	std::vector<std::vector<std::vector<T>>> v(a);
	for (size_t i = 0; i < a; ++i)
		v[i] = generate_random_2d<T>(b, c, min, max);
	return v;
}

// parameters order is assumed to be bfyx or oiyx
template<typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> generate_random_4d(size_t a, size_t b, size_t c, size_t d, int min, int max) {
	std::vector<std::vector<std::vector<std::vector<T>>>> v(a);
	for (size_t i = 0; i < a; ++i)
		v[i] = generate_random_3d<T>(b, c, d, min, max);
	return v;
}

// parameters order is assumed to be soiyx for filters when split > 1 
template<typename T>
std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>> generate_random_5d(size_t a, size_t b, size_t c, size_t d, size_t e, int min, int max) {
	std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>> v(a);
	for (size_t i = 0; i < a; ++i)
		v[i] = generate_random_4d<T>(b, c, d, e, min, max);
	return v;
}

template <class T> void set_value(const cldnn::pointer<T>& ptr, uint32_t index, T value) { ptr[index] = value; }
template <class T> T    get_value(const cldnn::pointer<T>& ptr, uint32_t index) { return ptr[index]; }

template<typename T>
void set_values(const cldnn::memory& mem, std::initializer_list<T> args ){
    auto ptr = mem.pointer<T>();

    auto it = ptr.begin();
    for(auto x : args)
        *it++ = x;
}

template<typename T>
void set_values(const cldnn::memory& mem, std::vector<T> args) {
    auto ptr = mem.pointer<T>();

    auto it = ptr.begin();
    for (auto x : args)
        *it++ = x;
}

template<typename T>
void set_random_values(const cldnn::memory& mem, float min, float max) 
{
	auto ptr = mem.pointer<T>();

	std::default_random_engine generator(1);
	std::uniform_real_distribution<float> distribution(min, max);

	for (auto it = ptr.begin(); it != ptr.end(); ++it)
	{
		*it = distribution(generator);
	}
}

// todo remove
// [[deprecated]]
template <typename T>
bool values_comparison(T first, T second, T threshold) {

    if (first == second) return true;

    auto abs_first = std::abs(first);
    auto abs_second = std::abs(second);
    auto delta = std::abs(abs_first - abs_second);

    if (abs_first == 0 || abs_second == 0 || delta < std::numeric_limits<T>::min()) {
        if (delta > threshold)
            return false;
    } else if ((delta / abs_first) > threshold)
        return false;

    return true;
}

// Checks equality of floats.
// For values less than absoulte_error_limit, absolute error will be counted
// for others, the relatve error will be counted.
// Function returns false if error will exceed the threshold.
// Default values:
// relative_error_threshold = 1e-3
// absolute_error_threshold = 1e-6
// absoulte_error_limit = 1e-4
inline bool are_equal(
    const float ref_item,
    const float item,
    const float relative_error_threshold = 1e-3,
    const float absolute_error_threshold = 1e-6,
    const float absoulte_error_limit     = 1e-4) {

        if( fabs(item) < absoulte_error_limit) {
            if(fabs( item - ref_item ) > absolute_error_threshold) {
                std::cout << "Ref val: " << ref_item << "\tSecond val: " << item << std::endl;
                return false;
            }
        } else
            if(fabs(item - ref_item) / fabs(ref_item) > relative_error_threshold){
                std::cout << "Ref val: " << ref_item << "\tSecond val: " << item << std::endl;
                return false;
        }

        return true;
}

inline bool floating_point_equal(FLOAT16 x, FLOAT16 y, int16_t max_ulps_diff = 4) {
	int16_t sign_bit_mask = 1;
	sign_bit_mask <<= 15;
	int16_t a = x.v, b = y.v;
	if ((a & sign_bit_mask) != (b & sign_bit_mask)) {
		a &= ~sign_bit_mask;
		b &= ~sign_bit_mask;
		return a == 0 && b == 0;
	}
	else {
		return std::abs(a - b) < max_ulps_diff;
	}
}

inline bool floating_point_equal(float x, float y, int32_t max_ulps_diff = 4) {
	int32_t sign_bit_mask = 1;
	sign_bit_mask <<= 31;
	int32_t a = reinterpret_cast<int32_t&>(x), b = reinterpret_cast<int32_t&>(y);
	if ((a & sign_bit_mask) != (b & sign_bit_mask)) {
		a &= ~sign_bit_mask;
		b &= ~sign_bit_mask;
		return a == 0 && b == 0;
	}
	else {
		return std::abs(a - b) < max_ulps_diff;
	}
}


class test_params
{
public:

	test_params(int32_t batch_size, int32_t feature_size, cldnn::tensor input_size) :
		input({ cldnn::format::bfyx,{ batch_size, feature_size, input_size.spatial[1],  input_size.spatial[0] } })
	{ }

	cldnn::tensor input;

	//TODO:
	//formats
	//data-types
	//input + output padding
};

class generic_test : public ::testing::TestWithParam<std::tuple<test_params*, cldnn::primitive*>>
{
	
public:

	generic_test();

	static void TearDownTestCase();

	void run_single_test();

	static std::vector<test_params*> generate_generic_test_params();

	virtual void generate_reference(cldnn::memory& input, cldnn::memory& output) = 0;

	struct custom_param_name_functor {
		std::string operator()(const ::testing::TestParamInfo<std::tuple<test_params*, cldnn::primitive*>>& info) {
			return std::to_string(info.index);
		}
	};

protected:

	test_params* generic_params;
	cldnn::primitive* layer_parmas;
	static std::vector<test_params*> all_generic_params;
};




}
