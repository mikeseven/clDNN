#pragma once
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
#pragma once
#include "neural.h"
#include <random>

namespace neural {

template <class T> T* data_begin(const memory &that) {return reinterpret_cast<T*>(that.pointer);}
template <class T> T* data_end(const memory &that)   {return data_begin<T>(that) + that.count();}

template <class T> void set_value(const memory &that, uint32_t index, T value) {data_begin<T>(that)[index] = value;}
template <class T> void set_value(const primitive &that, uint32_t index, T value) {set_value(that.as<const memory&>(), index, value);}
template <class T> T    get_value(const memory &that, uint32_t index)          {return data_begin<T>(that)[index];}

template <class T> void fill(const memory &that, T value) {
    if(type_id<T>()->id != memory::traits(that.argument.format).type->id) throw std::runtime_error("fill_memory: types do not match");
    std::fill(data_begin<T>(that), data_end<T>(that), value);
}
template <class T> void fill(const primitive &that, T value){
    fill(that.as<const memory&>(), value);
}

template <class T, class RNG = std::mt19937> void fill_rng(const memory &that, uint32_t seed, T dist_start, T dist_end) {
    if(type_id<T>()->id != memory::traits(that.argument.format).type->id) throw std::runtime_error("fill_memory: types do not match");
    static RNG rng(seed);
    std::uniform_real_distribution<T> dist(dist_start, dist_end);
    for(auto it1 = data_begin<T>(that), it2 = data_end<T>(that); it1 != it2; ++it1) *it1 = static_cast<T>( dist(rng) );
}
template <class T> void fill(const primitive &that){
    fill<float>(that.as<const memory&>());
}
}
} // namespace neural
