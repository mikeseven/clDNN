﻿/*
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
#include <numeric>
#include <mutex>
#include "api/neural.h"

namespace neural { namespace gpu {

struct neural_memory;
struct neural_vector;


class vector_arg {
    const neural::vector<uint32_t>& _vec;
    cl::Buffer _clBuffer;
public:
    vector_arg(const neural::vector<uint32_t>& arg);
    const cl::Buffer& get_buffer() const { return _clBuffer; };

    ~vector_arg();
};


class memory_arg {
    const neural::memory& _mem;
    cl::Buffer _clBuffer;
    bool is_own() const {
        return _mem.argument.engine == neural::engine::gpu && _mem.argument.owns_memory;
    }
    bool _copy_input;
    bool _copy_output;

protected:
    memory_arg(const neural::memory& mem, bool copy_input, bool copy_output);

public:
    const cl::Buffer& get_buffer() const { return _clBuffer; };
    ~memory_arg();
};

class input_mem : public memory_arg {
public:
    input_mem(const neural::memory& mem) :memory_arg(mem, true, false) {}
};

class output_mem : public memory_arg {
public:
    output_mem(const neural::memory& mem) :memory_arg(mem, false, true) {}
};

template<typename T, class Enable=void>
struct kernel_arg_handler;

template<typename T>
struct kernel_arg_handler<T, typename std::enable_if<!std::is_base_of<memory_arg, T>::value>::type> {
    static const T& get(const T& arg) { return arg; }
};

template<typename T>
struct kernel_arg_handler<T, typename std::enable_if<std::is_base_of<memory_arg, T>::value>::type> {
    static const cl::Buffer& get(const T& arg) { return arg.get_buffer(); }
};

template<>
struct kernel_arg_handler<vector_arg> {
    static const cl::Buffer& get(const vector_arg& arg) { return arg.get_buffer(); };
};


class kernel_execution_options {
    cl::NDRange _global;
    cl::NDRange _local;
public:
    kernel_execution_options(size_t work_items, size_t parallel_items) : _global(work_items), _local(parallel_items) {}

    cl::NDRange global_range() const { return _global; }
    cl::NDRange local_range() const { return _local; }
};

template<typename... Args>
class kernel {
    cl::Kernel _kernel;

    template<unsigned index, typename Ti, typename... Ts>
    void setArgs(Ti&& arg, Ts&&... args) {
        _kernel.setArg(index, kernel_arg_handler<Ti>::get(arg));
        setArgs<index + 1, Ts...>(std::forward<Ts>(args)...);
    }


    template<unsigned index, typename Ti>
    void setArgs(Ti&& arg) {
        _kernel.setArg(index, kernel_arg_handler<Ti>::get(arg));
    }

    template<unsigned index>
    void setArgs() {}

public:
    kernel(cl::Kernel kernel) : _kernel(kernel) {};
    kernel(const std::string& name);
    
    void operator()(const kernel_execution_options& options, Args... args)
    {
        setArgs<0>(std::forward<Args>(args)...);
        auto queue = cl::CommandQueue::getDefault();

        cl::Event end_event;
        queue.enqueueNDRangeKernel(_kernel, cl::NullRange, options.global_range(), options.local_range(), 0, &end_event);
        end_event.wait();
    }
};

template<typename Key, typename Type, class Traits = std::less<Key>,
    class Allocator = std::allocator<std::pair <const Key, Type> >>
class push_pop_map {
    std::mutex _mutex;
    std::map<Key, Type, Traits, Allocator> _map;
public:
    void push(const Key& key, Type value) {
        std::lock_guard<std::mutex> lock{ _mutex };
        _map.insert({ key, value });
    }

    Type pop(const Key& key) {
        std::lock_guard<std::mutex> lock{ _mutex };
        auto it = _map.find(key);
        if (it == _map.end()) throw std::out_of_range("Invalud push_pop_map<K, T> key");
        auto x = std::move(it->second);
        _map.erase(it);
        return std::move(x);
    }

    bool empty() {
        std::lock_guard<std::mutex> lock{ _mutex };
        return _map.empty();
    }
};

class gpu_toolkit {
    bool program_modified = true;
    cl::Program::Sources kernel_sources;

    std::unique_ptr<cl::Program> program;

    push_pop_map<void*, std::pair<const cl::Buffer, neural_memory*>> _mapped_memory;

    gpu_toolkit();
    
    ~gpu_toolkit() {
        assert(_mapped_memory.empty() && "There are mapped OCL buffers kept! Check the client code.");
    }

    static std::once_flag ocl_initialized;
    static void initialize();

    const cl::Program& get_program() {
        std::call_once(ocl_initialized, initialize);
        if (!program || program_modified) {
            program.reset(new cl::Program(kernel_sources));
            program->build();// "-cl-std=CL2.0");
            program_modified = false;
        }
        return *program.get();
    }

public:
    void add_kernel(const std::string& source) {
        kernel_sources.push_back(source);
        program_modified = true;
    }

    cl::Kernel get_kernel(const std::string& name) { return cl::Kernel(get_program(), name.c_str()); }

    neural_memory* new_memory_buffer(neural::memory::arguments arg);
    neural_memory* map_memory_buffer(const cl::Buffer& buf, cl::size_type size, cl_map_flags flags = CL_MAP_WRITE);
    cl::Buffer unmap_buffer(void* pointer);

    static gpu_toolkit& get();
};

template <typename ... Args>
kernel<Args...>::kernel(const std::string& name) : _kernel(gpu_toolkit::get().get_kernel(name)) {}

}}
