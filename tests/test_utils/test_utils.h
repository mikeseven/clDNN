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

template<typename T>
void fill( neural::primitive& prim, T val ){
    auto& mem = prim.as<const neural::memory&>();

    mem.fill(val);
}

template<typename T>
void fill( neural::primitive& prim ){
    auto& mem = prim.as<const neural::memory&>();

    mem.fill<T>();
}
}