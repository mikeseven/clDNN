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
#include "multidimensional_counter.h"

namespace ndimensional{

// This template has no body. it has to be specialized for every format. Every specialization should be placed in this file.
// index<T> is used only in by one function in whole project 'choose_calculate_idx', which is defined in this file.
// Because of above reasons this template is placed in cpp file.
template<neural::memory::format::type FORMAT>
size_t index(std::vector<uint32_t> size, std::vector<uint32_t> pos);

template<>
size_t index<neural::memory::format::yxfb_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(
    [&]() -> bool {
    for(size_t i = 0; i < pos.size(); ++i)
        if(size[i] <= pos[i]) return false;

        return true;
    }() == true );
    assert(pos.size() == size.size());

    // strides for yxfb format
    // vectors v_size and stride use format: b, f, spatials(y,x...)
    return pos[0] + size[0] * (pos[1] + size[1]*(pos[3] + size[3] * pos[2]));
};
template<>
size_t index<neural::memory::format::xb_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(
    [&]() -> bool {
    for(size_t i = 0; i < pos.size(); ++i)
        if(size[i] <= pos[i]) return false;

        return true;
    }() == true );
    assert(pos.size() == size.size());

    // strides for yxfb format
    // vectors v_size and stride use format: b, f, spatials(x)
    return pos[0] + size[0]*pos[2];
};
// NOT TESTED!
template<>
size_t index<neural::memory::format::bxyf_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos) {
    assert(
        [&]() -> bool {
        for (size_t i = 0; i < pos.size(); ++i)
            if (size[i] <= pos[i]) return false;

        return true;
    }() == true);
    assert(pos.size() == size.size());

    return pos[1] + size[1] * (pos[2] + size[2] * (pos[3] + size[3] * pos[0]));
};
template<>
size_t index<neural::memory::format::bfxy_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos) {
    assert(
        [&]() -> bool {
        for (size_t i = 0; i < pos.size(); ++i)
            if (size[i] <= pos[i]) return false;

        return true;
    }() == true);
    assert(pos.size() == size.size());

    return pos[2] + size[2] * (pos[3] + size[3] * (pos[1] + size[1] * pos[0]));
};
template<>
size_t index<neural::memory::format::oiyx_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(
        [&]() -> bool {
        for (size_t i = 0; i < pos.size(); ++i)
            if (size[i] <= pos[i]) return false;

        return true;
    }() == true);
    assert(pos.size() == size.size());
    assert(1 == size[0]); // batch

    return pos[4] + size[4] * (pos[3] + size[3] * (pos[2] + size[2] * pos[1]));
};
fptr choose_calculate_idx(neural::memory::format::type arg){
    fptr ptr;
    switch (arg){
        case neural::memory::format::type::x_f32: // treat x_f32 as xb_f32 with b=1
        case neural::memory::format::type::xb_f32:
            ptr = index<neural::memory::format::type::xb_f32>;
            break;
        case neural::memory::format::type::yxfb_f32:
            ptr = index<neural::memory::format::type::yxfb_f32>;
            break;
        case neural::memory::format::type::bxyf_f32:
            ptr = index<neural::memory::format::type::bxyf_f32>;
            break;
        case neural::memory::format::type::bfxy_f32:
            ptr = index<neural::memory::format::type::bfxy_f32>;
            break;
        case neural::memory::format::type::oiyx_f32:
            ptr = index<neural::memory::format::type::oiyx_f32>;
            break;
        default:
            throw std::runtime_error("choose_calculate_idx has no case for memory::format " + std::to_string(arg));
    }
    return ptr;
};

}
