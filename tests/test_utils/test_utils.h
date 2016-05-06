//todo move to another folder

#pragma once

#include <random>
#include "api/neural.h"
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
}