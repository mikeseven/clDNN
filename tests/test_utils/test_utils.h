//todo move to another folder

#pragma once

#include <random>
#include "api/neural.h"
namespace tests{

//todo remove
template<typename T>
void set_values_obsolete( neural::primitive& prim, std::initializer_list<T> args ){
    auto& mem = prim.as<const neural::memory_obselote&>();

    auto it = static_cast<T*>(mem.pointer);
    for(auto x : args)
        *it++ = x;
}

//todo remove
template<typename T>
void fill( neural::primitive& prim, T val ){
    auto& mem = prim.as<const neural::memory_obselote&>();

    mem.fill(val);
}

//todo remove
template<typename T>
void fill( neural::primitive& prim ){
    auto& mem = prim.as<const neural::memory_obselote&>();

    mem.fill<T>();
}

template<typename T>
void set_values( neural::primitive& prim, std::initializer_list<T> args ){
    auto& mem = prim.as<const neural::memory&>();

    auto it = static_cast<T*>(mem.pointer);
    for(auto x : args)
        *it++ = x;
}
}