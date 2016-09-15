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
#include <memory>
#include <cassert>
#include <iterator>
#include <numeric>
#include "ocl_toolkit.h"
#include "api/neural.h"

#define BUFFER_ALIGNMENT 4096
#define CACHE_ALIGNMENT 64

namespace neural { namespace gpu {

inline size_t align_to(size_t size, size_t align) {
    return (size % align == 0) ? size : size - size % align + align;
}

inline size_t pad_to(size_t size, size_t align) {
    return (size % align == 0) ? 0 : align - size % align;
}

template<typename T>
T* allocate_aligned(size_t size, size_t align) {
    assert(sizeof(T) <= size);
    assert(alignof(T) <= align);
    return reinterpret_cast<T*>(_mm_malloc(align_to(size, align), align));
}

template<typename T>
void deallocate_aligned(T* ptr) {
    _mm_free(ptr);
}

#if defined(_SECURE_SCL) && (_SECURE_SCL > 0)
template<typename T>
stdext::checked_array_iterator<T*> arr_begin(T* buf, size_t count) {
    return stdext::make_checked_array_iterator(buf, count);
}

template<typename T>
stdext::checked_array_iterator<T*> arr_end(T* buf, size_t count) {
    return stdext::make_checked_array_iterator(buf, count, count);
}

#else
template<typename T>
T* arr_begin(T* buf, size_t) { return buf; }

template<typename T>
T* arr_end(T* buf, size_t count) { return buf + count; }
#endif

#pragma pack(push, 4)
struct neural_memory {
    cl_uint format;
    cl_uint feature_offset;
    cl_uint spatial_offset;
    cl_uint vector_size;
    cl_uint data_offset;
    cl_uint data[1];

    void* pointer() { return reinterpret_cast<void*>(&data[data_offset]); }
    cl_uint* raw_begin() { return &data[0]; }
    const cl_uint* raw_begin() const { return &data[0]; }
    size_t raw_size() const { return vector_size; }

    size_t data_size() const {
        return std::accumulate(arr_begin(raw_begin(), raw_size()),
            arr_end(raw_begin(), raw_size()),
            memory::traits(static_cast<memory::format::type>(format)).type->size,
            std::multiplies<size_t>()
        );
    }

    void initialize(const neural::memory::arguments& arg) {
        format = static_cast<cl_uint>(arg.format);
        feature_offset = static_cast<cl_uint>(arg.size.batch.size());
        spatial_offset = static_cast<cl_uint>(arg.size.batch.size() + arg.size.feature.size());
        vector_size = static_cast<cl_uint>(arg.size.raw.size());

        data_offset = static_cast<cl_uint>(get_data_offset(arg));

        std::copy(std::begin(arg.size.raw), std::end(arg.size.raw), arr_begin(raw_begin(), raw_size()));
    }

    static std::vector<cl_uint> create_header(const neural::memory::arguments& arg) {
        std::vector<cl_uint> result {
            static_cast<cl_uint>(arg.format),
            static_cast<cl_uint>(arg.size.batch.size()),
            static_cast<cl_uint>(arg.size.batch.size() + arg.size.feature.size()),
            static_cast<cl_uint>(arg.size.raw.size()),
            static_cast<cl_uint>(get_data_offset(arg)),
        };
        result.insert(result.end(), arg.size.raw.begin(), arg.size.raw.end());
        return result;
    }

    static size_t header_size() {
        return sizeof(neural_memory) - sizeof(neural_memory::data);
    }

    static size_t datasize(const neural::memory::arguments& arg) {
        return std::accumulate(arg.size.raw.begin(),
            arg.size.raw.end(),
            memory::traits(arg.format).type->size,
            std::multiplies<size_t>()
        );
    }

    static size_t get_data_offset(const neural::memory::arguments& arg) {
        auto header_and_raw_size = header_size() + arg.size.raw.size() * sizeof(cl_uint);
        auto padding = pad_to(header_and_raw_size, CACHE_ALIGNMENT) / sizeof(cl_uint);
        return arg.size.raw.size() + padding;
    }

    static size_t size_of_memory(const neural::memory::arguments& arg) {
        return header_size() + get_data_offset(arg) * sizeof(cl_uint) + datasize(arg);
    }

    size_t size() const {
        return  header_size() + data_offset * sizeof(cl_uint) + data_size();
    }

    void* operator new(size_t) = delete;
    void* operator new[](size_t) = delete;
    void operator delete(void*) = delete;
    void operator delete[](void*) = delete;
};

struct neural_vector {
    cl_uint feature_offset;
    cl_uint spatial_offset;
    cl_uint raw_size;
    cl_uint raw[1];

    cl_uint* raw_begin() { return raw; }
    const cl_uint* raw_begin() const { return raw; }
    cl_uint* raw_end() { return raw + raw_size; }
    const cl_uint* raw_end() const { return raw + raw_size; }

    void initialize(const neural::vector<uint32_t>& src) {
        feature_offset = static_cast<cl_uint>(src.batch.size());
        spatial_offset = static_cast<cl_uint>(src.batch.size() + src.feature.size());
        raw_size = static_cast<cl_uint>(src.raw.size());

        std::copy(std::begin(src.raw), std::end(src.raw), arr_begin(raw_begin(), raw_size));
    }

    static size_t header_size() {
        return sizeof(neural_vector) - sizeof(neural_vector::raw);
    }

    static size_t size_of_vector(const neural::vector<uint32_t>& src) {
        return header_size() + src.raw.size() * sizeof(cl_uint);
    }

    size_t size() const {
        return  header_size() + raw_size * sizeof(cl_uint);
    }
};

static_assert(std::is_pod<neural_memory>::value, "Please fix the neural::gpu::neural_memory structure");
static_assert(std::is_pod<neural_vector>::value, "Please fix the neural::gpu::neural_vector structure");

#pragma pack(pop)

template<typename T>
struct sizeof_traits {
    static size_t get(size_t count) { return sizeof(T) * count; }
};

template<>
struct sizeof_traits<neural_memory> {
    static size_t get(const neural::memory::arguments& arg) { return neural_memory::size_of_memory(arg); }
};

template<>
struct sizeof_traits<neural_vector> {
    static size_t get(const neural::vector<uint32_t>& arg) { return neural_vector::size_of_vector(arg); }
};

struct gpu_buffer : public memory::buffer, public context_holder {
    gpu_buffer(memory::arguments arg);
    void* lock() override;
    void release() override;
    void reset(void* ptr) override;
    size_t size() override { return _data_size; }
    const cl::Buffer& get_buffer() const {
        assert(0 == _ref_count);
        return _buffer;
    }
private:
    std::mutex _mutex;
    memory::arguments _argument;
    unsigned _ref_count;
    size_t _buffer_size;
    size_t _data_size;
    cl::Buffer _buffer;
    neural_memory* _mapped_ptr;
};

} }
