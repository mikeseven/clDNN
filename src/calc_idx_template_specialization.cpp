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

namespace {
    bool is_in_range(std::vector<uint32_t> &range, std::vector<uint32_t> &pos) {
        if(pos.size()!=range.size()) return false;
        for(size_t i = 0; i < pos.size(); ++i)
            if(pos[i] >= range[i]) return false;
        return true;
    }
}


template<> size_t index<neural::memory::format::yxfb_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(is_in_range(size, pos));
    return pos[0] + size[0] * (pos[1] + size[1]*(pos[2] + size[2] * pos[3]));
};


template<> size_t index<neural::memory::format::oiyx_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(is_in_range(size, pos));
    assert(1 == size[0]); // batch
    return pos[3] + size[3] * (pos[4] + size[4] * (pos[2] + size[2] * pos[1]));
};


template<> size_t index<neural::memory::format::os_yxi_sv16_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(is_in_range(size, pos));
    assert(1 == size[0]); // batch
    uint32_t slice_id = pos[1] / 16;
    uint32_t id_in_slice = pos[1] % 16;
    return id_in_slice + 16 * (pos[2] + size[2] * (pos[3] + size[3] * (pos[4] + slice_id * size[4])));
};


template<> size_t index<neural::memory::format::byxf_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos) {
    assert(is_in_range(size, pos));
    return pos[1] + size[1] * (pos[2] + size[2] * (pos[3] + size[3] * pos[0]));
};


template<> size_t index<neural::memory::format::xb_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(is_in_range(size, pos));
    return pos[0] + size[0]*pos[2];
};


template<> size_t index<neural::memory::format::bx_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(is_in_range(size, pos));
    return pos[2] + size[2]*pos[0];
};


template<> size_t index<neural::memory::format::bfyx_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos) {
    assert(is_in_range(size, pos));
    return pos[2] + size[2] * (pos[3] + size[3] * (pos[1] + size[1] * pos[0]));
};


template<> size_t index<neural::memory::format::fyxb_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(is_in_range(size, pos));
    return pos[0] + size[0] * (pos[2] + size[2]*(pos[3] + size[3] * pos[1]));
}


template<> size_t index<neural::memory::format::byxf_b24_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
    const size_t B = size[0];   const size_t F = size[1];   const size_t X = size[2];
    const size_t b =  pos[0];   const size_t f =  pos[1];   const size_t x =  pos[2];   const size_t y =  pos[3];

    assert(24 == B); //todo remove, what with b>23
    return b%24 + B * (f + F*(x + X*y));
}


template<> size_t index<neural::memory::format::yxoi_o4_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
    const size_t Fo = size[1];   const size_t Fi = size[2];   const size_t X = size[3];
    const size_t fo =  pos[1];   const size_t fi =  pos[2];   const size_t x =  pos[3];   const size_t y =  pos[4];
    const size_t slice_block = fo/4;    const size_t slice_element = fo%4;

    assert(1 == size[0]); // Weights doesnt use batch, but the field must exist.
    assert(size[3] == size[4]); // X == Y
    assert(0 == Fo%4);
    return slice_element + 4*(fi + Fi*(slice_block + Fo/4*(x + X*y)));
}


template<>
size_t index<neural::memory::format::bs_yxf_bv24_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos) {
    assert(
        [&]() -> bool {
        for (size_t i = 0; i < pos.size(); ++i)
            if (size[i] <= pos[i]) return false;

        return true;
    }() == true);
    assert(pos.size() == size.size());
    assert(size[0] % 24 == 0); // batch

    uint32_t slice_id = pos[0] / 24;
    uint32_t id_in_slice = pos[0] % 24;

    auto idx = id_in_slice + 24 * (pos[1] + size[1] * (pos[2] + size[2] * (pos[3] + slice_id * size[3])));
    return idx;
};
fptr choose_calculate_idx(neural::memory::format::type arg){
    switch (arg){
        case neural::memory::format::type::x_f32: // treat x_f32 as xb_f32 with b=1
        case neural::memory::format::type::xb_f32:             return index<neural::memory::format::type::xb_f32>;
        case neural::memory::format::type::bx_f32:             return index<neural::memory::format::type::bx_f32>;
        case neural::memory::format::type::yxfb_f32:           return index<neural::memory::format::type::yxfb_f32>;
        case neural::memory::format::type::byxf_f32:           return index<neural::memory::format::type::byxf_f32>;
        case neural::memory::format::type::oiyx_f32:           return index<neural::memory::format::type::oiyx_f32>;
        case neural::memory::format::type::os_yxi_sv16_f32:    return index<neural::memory::format::type::os_yxi_sv16_f32>;
        case neural::memory::format::type::bfyx_f32:           return index<neural::memory::format::type::bfyx_f32>;
        case neural::memory::format::type::fyxb_f32:           return index<neural::memory::format::type::fyxb_f32>;
        case neural::memory::format::type::bs_yxf_bv24_f32:    return index<neural::memory::format::type::bs_yxf_bv24_f32>;
        case neural::memory::format::type::byxf_b24_f32:       return index<neural::memory::format::type::byxf_b24_f32>;
        case neural::memory::format::type::yxoi_o4_f32:        return index<neural::memory::format::type::yxoi_o4_f32>;
            break;
        default:
            throw std::runtime_error("choose_calculate_idx has no case for memory::format " + std::to_string(arg));
    }
};

}
