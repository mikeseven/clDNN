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

namespace {
    bool is_in_range(const std::vector<uint32_t> &range, const std::vector<uint32_t> &pos) {
        if(pos.size()!=range.size())
            return false;
        for(size_t i = 0; i < pos.size(); ++i)
            if(pos[i] >= range[i])
                return false;
        return true;
    }
}

// This template has no body. it has to be specialized for every format. Every specialization should be placed in this file.
// pointer<T> is used only in by one function in whole project 'choose_calculate_ptr', which is defined in this file.
// Because of above reasons this template is placed in cpp file.
template<neural::memory::format::type FORMAT>
void* pointer(const neural::memory& mem, const std::vector<uint32_t>& pos);

template <> void* pointer<neural::memory::format::byxf_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
                                const size_t F = size[1];   const size_t X = size[2];   const size_t Y = size[3];
    const size_t b = pos[0];    const size_t f =  pos[1];   const size_t x =  pos[2];   const size_t y =  pos[3];

    auto index =  f + F * (x + X * (y + Y * b));
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
};

template <> void* pointer<neural::memory::format::xb_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
    const size_t B = size[0];
    const size_t b =  pos[0];   const size_t x = pos[2];

    auto index = b + B*x;
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
};

template <> void* pointer<neural::memory::format::bx_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
    const size_t X = size[2];
    const size_t b =  pos[0];   const size_t x = pos[2];

    auto index = x + X*b;
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
};

template <> void* pointer<neural::memory::format::bfyx_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
                                const size_t F = size[1];   const size_t X = size[2];   const size_t Y = size[3];
    const size_t b = pos[0];    const size_t f =  pos[1];   const size_t x =  pos[2];   const size_t y =  pos[3];

    auto index = x + X * (y + Y * (f + F * b));
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
};

template <> void* pointer<neural::memory::format::yxfb_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
    const size_t B = size[0];   const size_t F = size[1];   const size_t X = size[2];
    const size_t b =  pos[0];   const size_t f =  pos[1];   const size_t x =  pos[2];   const size_t y = pos[3];

    auto index = b + B * (f + F*(x + X * y));
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
};

template <> void* pointer<neural::memory::format::oiyx_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));
    assert(1 == size[0]); // batch

                                const size_t Fi = size[2];  const size_t X = size[3];   const size_t Y = size[4];
    const size_t Fo = pos[1];   const size_t fi =  pos[2];  const size_t x =  pos[3];   const size_t y =  pos[4];

    auto index = x + X * (y + Y * (fi + Fi * Fo));
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
};

template <> void* pointer<neural::memory::format::oyxi_o16_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));
    assert(1 == size[0]); // batch

    uint32_t slice_id = pos[1] / 16;
    uint32_t id_in_slice = pos[1] % 16;

    auto index = id_in_slice + 16 * (pos[2] + size[2] * (pos[3] + size[3] * (pos[4] + slice_id * size[4])));
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
};

template <> void* pointer<neural::memory::format::fyxb_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
    const size_t B = size[0];                               const size_t X = size[2];   const size_t Y = size[3];
    const size_t b =  pos[0];   const size_t f = pos[1];    const size_t x =  pos[2];   const size_t y =  pos[3];

    auto index = b + B * (x + X*(y + Y * f));
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
}

template <> void* pointer<neural::memory::format::yxoi_o4_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
    const size_t Fo = size[1];   const size_t Fi = size[2];   const size_t X = size[3];
    const size_t fo =  pos[1];   const size_t fi =  pos[2];   const size_t x =  pos[3];   const size_t y =  pos[4];
    const size_t slice_block = fo/4;    const size_t slice_element = fo%4;

    assert(1 == size[0]); // Weights doesnt use batch, but the field must exist.
    assert(size[3] == size[4]); // X == Y
    assert(0 == Fo%4);
    auto index = slice_element + 4*(fi + Fi*(slice_block + Fo/4*(x + X*y)));
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
}

template <> void* pointer<neural::memory::format::byxf_b24_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

                                const size_t F = size[1];   const size_t X = size[2];   const size_t Y = size[3];
    const size_t b =  pos[0];   const size_t f =  pos[1];   const size_t x =  pos[2];   const size_t y =  pos[3];

    assert(size[0]%24==0); // batch must be a multiple of 24
    auto index = b%24 + 24 * (f + F * (x + X * (y + (b/24) * Y)));
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
};

template <> void* pointer<neural::memory::format::oi_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
  /*  const size_t Fo = size[1];*/ const size_t Fi = size[2];
    const size_t fo =  pos[1];     const size_t fi =  pos[2];

    assert(1 == size[0]); // Weights doesnt use batch, but the field must exist.
    assert(1 == size[3]);

    auto index = fi + fo * Fi;
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
}

template <> void* pointer<neural::memory::format::io_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
    const size_t Fo = size[1];  // const size_t Fi = size[2];
    const size_t fo =  pos[1];   const size_t fi =  pos[2];

    assert(1 == size[0]); // Weights doesnt use batch, but the field must exist.
    assert(1 == size[3]);

    auto index = fo + fi * Fo;
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
}

template <> void* pointer<neural::memory::format::io_i13_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
 /*   const size_t Fo = size[1];*/   const size_t Fi = size[2];
    const size_t fo =  pos[1];       const size_t fi =  pos[2];

    const uint32_t stride = 13;
    assert(1 == size[0]); // Weights doesnt use batch, but the field must exist.
    assert(1 == size[3]);
    assert(0 == size[1] % stride );

    auto index = fo % stride + fi * stride +(stride * Fi)*(fo / stride);
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);

 }

template <> void* pointer<neural::memory::format::io_i2_f32>(const neural::memory& mem, const std::vector<uint32_t>& pos){
    auto& size = mem.argument.size.raw;
    assert(is_in_range(size, pos));

    // BFXY represents buffer size, wbile bfxy represents current position
  /*  const size_t Fo = size[1];*/  const size_t Fi = size[2];
    const size_t fo =  pos[1];      const size_t fi =  pos[2];

    const uint32_t stride = 2;
    assert(1 == size[0]); // Weights doesnt use batch, but the field must exist.
    assert(1 == size[3]);
    assert(0 == size[1] % stride );

    auto index = fo % stride + fi * stride +(stride * Fi)*(fo / stride);
    float* array_ptr = static_cast<float*>(mem.pointer);
    return &(array_ptr[index]);
}

pfptr choose_calculate_ptr(const neural::memory& mem)
{
    auto format = mem.argument.format;
    switch (format)
    {
        case neural::memory::format::type::x_f32: // treat x_f32 as xb_f32 with b=1
        case neural::memory::format::type::xb_f32:          return pointer<neural::memory::format::type::xb_f32>;
        case neural::memory::format::type::bx_f32:          return pointer<neural::memory::format::type::bx_f32>;
        case neural::memory::format::type::oi_f32:          return pointer<neural::memory::format::type::oi_f32>;
        case neural::memory::format::type::io_f32:          return pointer<neural::memory::format::type::io_f32>;
        case neural::memory::format::type::io_i13_f32:      return pointer<neural::memory::format::type::io_i13_f32>;
        case neural::memory::format::type::io_i2_f32:       return pointer<neural::memory::format::type::io_i2_f32>;
        case neural::memory::format::type::yxfb_f32:        return pointer<neural::memory::format::type::yxfb_f32>;
        case neural::memory::format::type::byxf_f32:        return pointer<neural::memory::format::type::byxf_f32>;
        case neural::memory::format::type::oiyx_f32:        return pointer<neural::memory::format::type::oiyx_f32>;
        case neural::memory::format::type::oyxi_o16_f32:    return pointer<neural::memory::format::type::oyxi_o16_f32>;
        case neural::memory::format::type::bfyx_f32:        return pointer<neural::memory::format::type::bfyx_f32>;
        case neural::memory::format::type::fyxb_f32:        return pointer<neural::memory::format::type::fyxb_f32>;
        case neural::memory::format::type::byxf_b24_f32:    return pointer<neural::memory::format::type::byxf_b24_f32>;
        case neural::memory::format::type::yxoi_o4_f32:     return pointer<neural::memory::format::type::yxoi_o4_f32>;
        default:
            throw std::runtime_error("choose_calculate_ptr has no case for memory::format " + std::to_string(format));
    }
}

}