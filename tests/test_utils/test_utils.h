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

namespace tests {

#define USE_RANDOM_SEED 0
#if USE_RANDOM_SEED
	std::random_device rnd_device;
	unsigned int const random_seed = rnd_device();
#else
	unsigned int const random_seed = 1337;
#endif

using VF = std::vector<float>;		// float vector
using VVF = std::vector<VF>;		// feature map
using VVVF = std::vector<VVF>;		// 3d feature map
using VVVVF = std::vector<VVVF>;	// batch of 3d feature maps
using VVVVVF = std::vector<VVVVF>;	// split of oiyx filters

inline VF flatten_4d(cldnn::format format, VVVVF &data) {
	size_t a = data.size();
	size_t b = data[0].size();
	size_t c = data[0][0].size();
	size_t d = data[0][0][0].size();
	VF vec(a * b * c * d, 0.0f);
	size_t idx = 0;
	if (format == cldnn::format::yxfb) {
		for (size_t yi = 0; yi < c; ++yi)
			for (size_t xi = 0; xi < d; ++xi)
				for (size_t fi = 0; fi < b; ++fi)
					for (size_t bi = 0; bi < a; ++bi)
						vec[idx++] = data[bi][fi][yi][xi];
	}
	else if (format == cldnn::format::oiyx) {
		for (size_t oi = 0; oi < a; ++oi)
			for (size_t ii = 0; ii < b; ++ii)
				for (size_t yi = 0; yi < c; ++yi)
					for (size_t xi = 0; xi < d; ++xi)
						vec[idx++] = data[oi][ii][yi][xi];
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

}
