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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cstdint>
#include "cldnn_defs.h"
#include "compounds.h"
#include "tensor.hpp"

namespace cldnn
{
#define FLOAT_TYPE_MASK 0x80000000

enum class data_types : uint32_t
{
    i8  = sizeof(int8_t),
    i16 = sizeof(int16_t),
    i32 = sizeof(int32_t),
    i64 = sizeof(int64_t),
    f16 = sizeof(int16_t) | FLOAT_TYPE_MASK,
    f32 = sizeof(float)   | FLOAT_TYPE_MASK,
    f64 = sizeof(double)  | FLOAT_TYPE_MASK,
};

struct data_type_traits
{
    static size_t size_of(data_types data_type)
    {
        return (static_cast<uint32_t>(data_type) & ~FLOAT_TYPE_MASK);
    }

    static bool is_floating_point(data_types data_type)
    {
        return (static_cast<uint32_t>(data_type) & FLOAT_TYPE_MASK) != 0;
    }
};

template <typename T>
bool data_type_match(data_types data_type)
{
    return (sizeof(T) == 1) || (sizeof(T) == data_type_traits::size_of(data_type));
}

template <typename T> struct data_type_selector;
template<> struct data_type_selector  <int8_t> { const data_types value = data_types::i8;  };
template<> struct data_type_selector <uint8_t> { const data_types value = data_types::i8;  };
template<> struct data_type_selector <int16_t> { const data_types value = data_types::i16; };
template<> struct data_type_selector<uint16_t> { const data_types value = data_types::i16; };
template<> struct data_type_selector <int32_t> { const data_types value = data_types::i32; };
template<> struct data_type_selector<uint32_t> { const data_types value = data_types::i32; };
template<> struct data_type_selector <int64_t> { const data_types value = data_types::i64; };
template<> struct data_type_selector<uint64_t> { const data_types value = data_types::i64; };
template<> struct data_type_selector  <half_t> { const data_types value = data_types::f16; };
template<> struct data_type_selector   <float> { const data_types value = data_types::f32; };
template<> struct data_type_selector  <double> { const data_types value = data_types::f64; };

struct layout
{
    layout(data_types data_type, tensor size)
        : data_type(data_type)
        , size(size)
    {}

    size_t data_size() const { return data_type_traits::size_of(data_type) * size.get_linear_size(); }
    size_t count() const { return size.get_linear_size(); }

    data_types data_type;
    tensor size;
};

struct buffer;

struct memory
{
    size_t count() const { return _layout.count(); }
    size_t size() const { return _layout.data_size(); }
    layout get_layout() const { return _layout; }
    DLL_SYM ~memory();

    DLL_SYM memory(const memory& other);
    DLL_SYM memory& operator=(const memory& other);
private:
    friend struct engine;
    memory(cldnn::layout l, buffer* d);
    layout _layout;
    buffer* _data;
    template<typename T> friend struct pointer;
    DLL_SYM void* lock();
    DLL_SYM void unlock();
};

static_assert(std::is_standard_layout<memory>::value, "class has to be 'standart layout'");

template<typename T>
struct pointer
{
    pointer(const memory& mem): _mem(mem), _ptr(lock_ptr(mem)){}
    ~pointer() { _mem.unlock(); }

    T* data() { return _ptr; }
    size_t size() const { return _mem.count(); }

#if defined(_SECURE_SCL) && (_SECURE_SCL > 0)
    typedef stdext::checked_array_iterator<T*> iterator;
    stdext::checked_array_iterator<T*> begin() const&
    {
        return stdext::make_checked_array_iterator(_ptr, size());
    }

    stdext::checked_array_iterator<T*> end() const&
    {
        return stdext::make_checked_array_iterator(_ptr, size(), size());
    }
#else
    typedef T* iterator;
    T* begin() const& { return _ptr; }
    T* end() const& { return _ptr + size(); }
#endif

    T& operator[](size_t idx) const&
    {
        assert(idx < size());
        return _ptr[idx];
    }

    friend bool operator==(const pointer& lhs, const pointer& rhs)
    {
        return lhs.data == rhs.data;
    }

    friend bool operator!=(const pointer& lhs, const pointer& rhs)
    {
        return !(lhs == rhs);
    }

    // do not use this class as temporary object
    // ReSharper disable CppMemberFunctionMayBeStatic, CppMemberFunctionMayBeConst
    void begin() && {}
    void end() && {}
    void operator[](size_t idx) && {}
    // ReSharper restore CppMemberFunctionMayBeConst, CppMemberFunctionMayBeStatic

private:
    static T* lock_ptr(memory& mem)
    {
        if (!data_type_match<T>(mem.layout.data_type)) throw std::logic_error("memory data type do not match");
        return static_cast<T*>(mem.lock());
    }

    memory _mem;
    T* _ptr;
};
}