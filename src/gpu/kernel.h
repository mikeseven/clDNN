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
#include "ocl_toolkit.h"
#include "kernels_cache.h"
#include <iostream>

namespace neural { namespace gpu {

class vector_arg : public context_holder {
    const neural::vector<uint32_t>& _vec;
    cl::Buffer _clBuffer;
public:
    vector_arg(const neural::vector<uint32_t>& arg);
    const cl::Buffer& get_buffer() const { return _clBuffer; };

    ~vector_arg();
};

class memory_arg : public context_holder {
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

template<typename T, class Enable = void>
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
class kernel : public context_holder {
    kernels_cache::kernel_id _kernel_id;
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
    explicit kernel(const std::string& name, std::map<std::string, std::string> definitions = std::map<std::string, std::string>()) : _kernel_id(kernels_cache::get().create_kernel_from_template(name, definitions)) {}

    void operator()(const kernel_execution_options& options, Args... args) {
        _kernel = kernels_cache::get().get_kernel(context().get(), _kernel_id);
        setArgs<0>(std::forward<Args>(args)...);

        try {
            cl::Event end_event;
            context()->queue().enqueueNDRangeKernel(_kernel, cl::NullRange, options.global_range(), options.local_range(), 0, &end_event);
            end_event.wait();
        } catch(cl::Error err) {
            std::cerr << "ERROR:" << err.what() << std::endl;
        }
    }
};

} }
