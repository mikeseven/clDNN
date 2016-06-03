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

#include "api/neural.h"
#include <iostream>
#include <limits>
#include <random>

namespace tests{

template<typename T>
void set_values( neural::primitive& prim, std::initializer_list<T> args ){
    auto& mem = prim.as<const neural::memory&>();

    auto it = static_cast<T*>(mem.pointer);
    for(auto x : args)
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

        float diff = fabs(item - ref_item);
        if( diff < absoulte_error_limit) {
            if( diff > absolute_error_threshold) {
                std::cout << "Ref val: " << ref_item << "\tSecond val: " << item << std::endl; //todo remove
                return false;
            }
        } else
            if( diff / fabs(ref_item) > relative_error_threshold){
                std::cout << "Ref val: " << ref_item << "\tSecond val: " << item << std::endl;
                return false;
        }

        return true;
}

}
