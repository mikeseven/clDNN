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
#include "engine.hpp"

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

// TODO remove this backward compatibility class
struct neural_memory
{
    struct format
    {
        enum type : uint8_t
        {
            // FP32 (single precision float)
            x_f32,
            xb_f32,     // 1D+batch, float32
            bx_f32,     // 1D+batch, float32
            yxfn_f32,   // 3D + number of neurons - used in fully connected weights
            yxfb_f32,   // 3D+batch, float32
            byxf_f32,   // for convolution_cpu_jit_batch1
            bfyx_f32,   // used in Caffe
            fyxb_f32,   // used in Caffe
            oiyx_f32,   // format used only for weights: o - output feature maps, i - input feature maps
            yxoi_f32,   // format used only for weights: o - output feature maps, i - input feature maps
            oyxi_f32,   // format used only for weights: o - output feature maps, i - input feature maps
            yxio_f32,   // format used only for weights: o - output feature maps, i - input feature maps

            // FP16 (half precision float)
            x_f16,
            xb_f16,            // 1D+batch, FP16 (half precision float)
            bx_f16,            // 1D+batch, FP16 (half precision float)
            yxfn_f16,          // 3D + number of neurons - used in fully connected weights
            yxfb_f16,          // 3D+batch, FP16 (half precision float)
            byxf_f16,          // for convolution_cpu_jit_batch1
            bfyx_f16,          // used in Caffe
            fyxb_f16,          // used in Caffe
            oiyx_f16,          // format used only for weights: o - output feature maps, i - input feature maps
            yxoi_f16,          // format used only for weights: o - output feature maps, i - input feature maps
            oyxi_f16,          // format used only for weights: o - output feature maps, i - input feature maps
            yxio_f16,          // format used only for weights: o - output feature maps, i - input feature maps

            format_num,
            any = static_cast<uint8_t>(-1),
            half_base = x_f16
        };
    };

    struct type_traits {
        type_traits(cldnn::data_types data_type)
            : size(cldnn::data_type_traits::size_of(data_type))
            , is_floating_point(cldnn::data_type_traits::is_floating_point(data_type))
        {}
        const size_t          size;
        const bool            is_floating_point;
    };

    struct format_traits
    {
        format_traits(uint8_t dimension, cldnn::data_types data_type)
            : dimension(dimension)
            , type(new type_traits(data_type))
        {
        }
        const uint8_t       dimension;
        std::unique_ptr<type_traits>  type;
    };

    static format_traits traits(const cldnn::layout& layout)
    {
        return format_traits(layout.size.format.order().length(), layout.data_type);
    }

    static uint8_t get_format_base(cldnn::format format)
    {
        switch (format.value)
        {
        case cldnn::format::x:    return format::type::x_f32;
        case cldnn::format::xb:   return format::type::xb_f32;
        case cldnn::format::bx:   return format::type::bx_f32;
        case cldnn::format::yxfn: return format::type::yxfn_f32;
        case cldnn::format::yxfb: return format::type::yxfb_f32;
        case cldnn::format::byxf: return format::type::byxf_f32;
        case cldnn::format::bfyx: return format::type::bfyx_f32;
        case cldnn::format::fyxb: return format::type::fyxb_f32;
        case cldnn::format::oiyx: return format::type::oiyx_f32;
        case cldnn::format::yxoi: return format::type::yxoi_f32;
        case cldnn::format::oyxi: return format::type::oyxi_f32;
        case cldnn::format::yxio: return format::type::yxio_f32;
        default: throw std::invalid_argument("unsupported format");
        }
    }

    static format::type convert_format(const cldnn::layout& layout)
    {
        switch (layout.size.format.value)
        {
        case cldnn::format::format_num: return format::type::format_num;
        case cldnn::format::any: return format::type::any;
        default: break;
        }

        uint8_t format_shift;
        switch (layout.data_type)
        {
        case cldnn::data_types::f32:
            format_shift = 0;
            break;
        case cldnn::data_types::f16:
            format_shift = format::type::half_base;
            break;
        default: throw std::invalid_argument("unsupported data type");
        }
        return static_cast<format::type>(get_format_base(layout.size.format) + format_shift);
    }

    struct arguments {
        format::type    format;
        const tensor&   size;
        arguments(const cldnn::layout& layout): format(convert_format(layout)), size(layout.size){}
    };
};

struct memory_impl;
template<typename T> struct pointer;
struct memory
{
    static memory allocate(const engine& engine, const layout& layout)
    {
        size_t size = layout.data_size();
        if (size == 0) throw std::invalid_argument("size should be more than 0");
        status_t status;
        auto buf = allocate_buffer(engine, layout, &status);
        if (buf == nullptr || status != CLDNN_SUCCESS)
            CLDNN_THROW("memory allocation failed", status);
        return memory(buf);
    }

    template<typename T>
    static memory attach(const cldnn::layout& layout, array_ref<T> array)
    {
        if (array.empty()) throw std::invalid_argument("array should not be empty");
        size_t size = array.size() * sizeof(T);
        if ( size != layout.data_size()) throw std::invalid_argument("buffer size mismatch");
        status_t status;
        auto buf = attach_buffer(layout, array.data(), size, &status);
        if (buf == nullptr || status != CLDNN_SUCCESS)
            CLDNN_THROW("memory attach failed", status);
        return memory(buf);
    }

    memory(memory_impl* data)
        :_data(data)
    {
        if (!_data) throw std::invalid_argument("data");
    }

    DLL_SYM memory(const memory& other);

    DLL_SYM memory& operator=(const memory& other);

    DLL_SYM ~memory();

    friend bool operator==(const memory& lhs, const memory& rhs)
    {
        return lhs._data == rhs._data;
    }

    friend bool operator!=(const memory& lhs, const memory& rhs)
    {
        return !(lhs == rhs);
    }

    /**
     * \brief 
     * \return number of elements of _layout.data_type stored in memory
     */
    size_t count() const { return get_layout().count(); }

    /**
     * \brief 
     * \return number of bytes used by memory
     */
    size_t size() const { return get_layout().data_size(); }
    DLL_SYM const layout& get_layout() const noexcept;
    DLL_SYM bool is_allocated_by(const engine& engine) const noexcept;

    // TODO remove this backward compatibility call
    neural_memory::arguments argument() const { return neural_memory::arguments(get_layout()); };
    template<typename T> pointer<T> pointer() const;

    const memory_impl* get() const { return _data; }

private:
    friend struct engine;
    memory_impl* _data;
    DLL_SYM static memory_impl* allocate_buffer(engine engine, layout layout, status_t* status) noexcept;
    DLL_SYM static memory_impl* attach_buffer(layout layout, void* pointer, size_t size, status_t* status) noexcept;
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
        if (!data_type_match<T>(_mem.get_layout().data_type)) throw std::logic_error("memory data type do not match");
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

template <typename T>
pointer<T> memory::pointer() const { return cldnn::pointer<T>(*this); }

}
