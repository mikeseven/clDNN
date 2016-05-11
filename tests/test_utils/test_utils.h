//todo move to another folder

#pragma once

#include <random>
#include "api/neural.h"
#include <limits>

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
    const float item,
    const float ref_item,
    const float relative_error_threshold = 1e-3,
    const float absolute_error_threshold = 1e-6,
    const float absoulte_error_limit     = 1e-4) {

        if( fabs(item) < absoulte_error_limit) {
            if(fabs( item - ref_item ) > absolute_error_threshold) {
                return false;
            }
        } else
            if(fabs(item - ref_item) / fabs(ref_item) > relative_error_threshold)
                return false;

        return true;
}

}
