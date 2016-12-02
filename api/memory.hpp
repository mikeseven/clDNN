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

    layout(const layout& other)
        : data_type(other.data_type)
        , size(other.size)
    {
    }

    layout& operator=(const layout& other)
    {
        if (this == &other)
            return *this;
        data_type = other.data_type;
        size = other.size;
        return *this;
    }

    friend bool operator==(const layout& lhs, const layout& rhs)
    {
        return lhs.data_type == rhs.data_type
                && lhs.size == rhs.size;
    }

    friend bool operator!=(const layout& lhs, const layout& rhs)
    {
        return !(lhs == rhs);
    }

    /**
     * \brief 
     * \return number of bytes needed to store this layout
     */
    size_t data_size() const { return data_type_traits::size_of(data_type) * size.get_linear_size(); }

    /**
     * \brief 
     * \return number of elements to be stored in this layout
     */
    size_t count() const { return size.get_linear_size(); }

    data_types data_type;
    tensor size;
};

struct memory_impl;
struct memory
{
    static memory allocate(const cldnn::layout& layout)
    {
        size_t size = layout.data_size();
        if (size == 0) throw std::invalid_argument("size should be more than 0");
        status_t status;
        auto buf = allocate_buffer(size, &status);
        if (buf == nullptr || status != CLDNN_SUCCESS)
            CLDNN_THROW("memory allocation failed", status);
        return memory(layout, buf);
    }

    template<typename T>
    static memory attach(const cldnn::layout& layout, array_ref<T> array)
    {
        if (array.empty()) throw std::invalid_argument("array should not be empty");
        size_t size = array.size() * sizeof(T);
        if ( size != layout.data_size()) throw std::invalid_argument("buffer size mismatch");
        status_t status;
        auto buf = attach_buffer(array.data(), size, &status);
        if (buf == nullptr || status != CLDNN_SUCCESS)
            CLDNN_THROW("memory attach failed", status);
        return memory(layout, buf);
    }

    static memory attach(const cldnn::layout& layout, void* pointer, size_t size)
    {
        if (pointer == nullptr) throw std::invalid_argument("pointer should not be NULL");
        if (size == 0) throw std::invalid_argument("size should be more than 0");
        if (size < layout.data_size()) throw std::invalid_argument("buffer size mismatch");
        status_t status;
        auto buf = attach_buffer(pointer, size, &status);
        if (buf == nullptr || status != CLDNN_SUCCESS)
            CLDNN_THROW("memory attach failed", status);
        return memory(layout, buf);
    }

    DLL_SYM memory(const memory& other);

    DLL_SYM memory& operator=(const memory& other);

    DLL_SYM ~memory();

    friend bool operator==(const memory& lhs, const memory& rhs)
    {
        return lhs._layout == rhs._layout
                && lhs._data == rhs._data;
    }

    friend bool operator!=(const memory& lhs, const memory& rhs)
    {
        return !(lhs == rhs);
    }

    /**
     * \brief 
     * \return number of elements of _layout.data_type stored in memory
     */
    size_t count() const { return _layout.count(); }

    /**
     * \brief 
     * \return number of bytes used by memory
     */
    size_t size() const { return _layout.data_size(); }
    layout get_layout() const { return _layout; }

private:
    memory(const cldnn::layout& layout, memory_impl* data)
        :_layout(layout), _data(data)
    {
        if (!_data) throw std::invalid_argument("data");
    }
    friend struct engine;
    layout _layout;
    memory_impl* _data;
    DLL_SYM static memory_impl* allocate_buffer(size_t size, status_t* status) noexcept;
    DLL_SYM static memory_impl* attach_buffer(void* pointer, size_t size, status_t* status) noexcept;
    DLL_SYM void* lock_buffer(status_t* status) const noexcept;
    DLL_SYM status_t unlock_buffer() const noexcept;

    void* lock() const
    {
        status_t status;
        auto ptr = lock_buffer(&status);
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("memory lock failed", status);
        return ptr;
    }

    void unlock() const
    {
        status_t status = unlock_buffer();
        if (status != CLDNN_SUCCESS)
            CLDNN_THROW("memory unlock failed", status);
    }
    template<typename T> friend struct pointer;
};

API_CLASS(memory)

template<typename T>
struct pointer
{
    pointer(const memory& mem): _mem(mem)
    {
        if (!data_type_match<T>(_mem._layout.data_type)) throw std::logic_error("memory data type do not match");
        _ptr = static_cast<T*>(_mem.lock());
    }
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
    memory _mem;
    T* _ptr;
};
}
