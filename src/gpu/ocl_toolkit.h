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

#pragma warning(push)
#pragma warning(disable: 4100)
#pragma warning(disable: 4505)
// we want exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "cl2.hpp"
#pragma warning(pop)
#include <mutex>
#include "api/neural.h"
#include <atomic>
#include <sstream>
#include "push_pop_map.h"
#include "memory_gpu.h"

namespace neural { namespace gpu {
class gpu_toolkit;

struct neural_memory;
struct neural_vector;

template<typename T>
struct sizeof_traits {
    static size_t get(size_t count) { return sizeof(T) * count ; }
};

template<>
struct sizeof_traits<neural_memory> {
    static size_t get(const neural::memory::arguments& arg) { return neural_memory::size_of_memory(arg); }
};

template<>
struct sizeof_traits<neural_vector> {
    static size_t get(const neural::vector<uint32_t>& arg) { return neural_vector::size_of_vector(arg); }
};

template<typename T, typename Size_of = sizeof_traits<T>>
class mapped_buffer {
    std::shared_ptr<gpu_toolkit> _context;
    cl::Buffer _buffer;
    T* _mapped_ptr;
public:
    template<typename Arg> mapped_buffer(std::shared_ptr<gpu_toolkit> context, cl::Buffer buffer, Arg arg);
    template<typename Arg> mapped_buffer(std::shared_ptr<gpu_toolkit> context, Arg arg);
    ~mapped_buffer();

    mapped_buffer(const mapped_buffer& other) = delete;
    mapped_buffer& operator=(const mapped_buffer& other) = delete;

    T* data() const { return _mapped_ptr; }
    cl::Buffer buffer() const { return _buffer; }
};

class gpu_toolkit {
    cl::Device _device;
    cl::Context _context;
    cl::CommandQueue _command_queue;
    cl::Program _program;

    gpu_toolkit();
    
    ~gpu_toolkit() {
        assert(_mapped_memory.empty() && "There are mapped OCL buffers kept! Check the client code.");
    }

    static std::shared_ptr<gpu_toolkit>get();
    friend class context_holder;
public:

    push_pop_map<void*, std::unique_ptr<mapped_buffer<neural_memory>>> _mapped_memory;

    mapped_buffer<neural_memory>* new_memory_buffer(neural::memory::arguments arg);
    mapped_buffer<neural_memory>* map_memory_buffer(const cl::Buffer& buf, neural::memory::arguments arg);
    std::unique_ptr<mapped_buffer<neural_memory>> unmap_buffer(void* pointer);

    cl::Device& device() { return _device; }
    cl::Context& context() { return _context; }
    cl::CommandQueue& queue() { return _command_queue; }
    cl::Program& program() { return _program; }

    static void* allocate_memory_gpu(neural::memory::arguments arg);
    static void deallocate_memory_gpu(void* pointer, neural::memory::arguments);

    gpu_toolkit(const gpu_toolkit& other) = delete;
    gpu_toolkit(gpu_toolkit&& other) = delete;
    gpu_toolkit& operator=(const gpu_toolkit& other) = delete;
    gpu_toolkit& operator=(gpu_toolkit&& other) = delete;
};

class context_holder {
    std::shared_ptr<gpu_toolkit> _context;
protected:
    context_holder() : _context(gpu_toolkit::get()){}
    virtual ~context_holder() = default;
    const std::shared_ptr<gpu_toolkit>& context() const { return _context; }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename Size_of>
template <typename Arg>
mapped_buffer<T, Size_of>::mapped_buffer(std::shared_ptr<gpu_toolkit> context, cl::Buffer buffer, Arg arg) :
    _context(context), _buffer(buffer) {
    cl::Event end_event;
    _mapped_ptr = reinterpret_cast<T*>(_context->queue().enqueueMapBuffer(_buffer, true, CL_MAP_WRITE, 0, Size_of::get(arg), 0, &end_event));
    end_event.wait();
}

template <typename T, typename Size_of>
template <typename Arg>
mapped_buffer<T, Size_of>::mapped_buffer(std::shared_ptr<gpu_toolkit> context, Arg arg) :
    mapped_buffer(context, cl::Buffer(context->context(), CL_MEM_READ_WRITE, Size_of::get(arg)), arg) {}

template <typename T, typename Size_of>
mapped_buffer<T, Size_of>::~mapped_buffer() {
    cl::Event end_event;
    _context->queue().enqueueUnmapMemObject(_buffer, _mapped_ptr, 0, &end_event);
    end_event.wait();
}

}}
