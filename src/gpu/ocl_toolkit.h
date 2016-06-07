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
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include "cl2.hpp"
#pragma warning(pop)
#include <numeric>
#include <mutex>
#include "api/neural.h"

namespace neural { namespace gpu {

struct neural_memory;

class buffer {
    const neural::memory& _mem;
    cl::Buffer _clBuffer;
    bool is_own() const {
        return _mem.argument.engine == neural::engine::gpu && _mem.argument.owns_memory;
    }
    bool _copy_input;
    bool _copy_output;

    size_t size() const;

protected:
    buffer(const neural::memory& mem, bool copy_input, bool copy_output);

public:
    cl::Buffer get_buffer() const { return _clBuffer; }

    ~buffer();
};

class input_mem : public buffer {
public:
    input_mem(const neural::memory& mem) :buffer(mem, true, false) {}
};

class output_mem : public buffer {
public:
    output_mem(const neural::memory& mem) :buffer(mem, false, true) {}
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

    
    template<typename Ti>
    void setKernelArg(unsigned index, const Ti& arg) {
        _kernel.setArg(index, arg);
    }

    void setKernelArg(unsigned index, const input_mem& arg) {
        _kernel.setArg(index, arg.get_buffer());
    }

    void setKernelArg(unsigned index, const output_mem& arg) {
        _kernel.setArg(index, arg.get_buffer());
    }

    template<unsigned index, typename Ti, typename... Ts>
    void setArgs(Ti&& arg, Ts&&... args) {
        setKernelArg(index, arg);
        setArgs<index + 1, Ts...>(std::forward<Ts>(args)...);
    }


    template<unsigned index, typename Ti>
    void setArgs(Ti&& arg) {
        setKernelArg(index, arg);
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

class gpu_toolkit {
public:
    using buffer_type = cl::Buffer;

private:

    bool program_modified = true;
    cl::Program::Sources kernel_sources;

    std::unique_ptr<cl::Program> program;

    std::map<void*, std::pair<const cl::Buffer, neural_memory*>> _mapped_buffers;

private:
    gpu_toolkit();

    static std::once_flag ocl_initialized;
    static void initialize();

    const cl::Program& get_program() {
        if (!program || program_modified) {
            program.reset(new cl::Program(kernel_sources));
            program->build("-cl-std=CL2.0");
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

    neural_memory* new_buffer(neural::memory::arguments arg);
    neural_memory* map_buffer(const cl::Buffer& buf, cl::size_type size, cl_map_flags flags = CL_MAP_WRITE);
    cl::Buffer unmap_buffer(void* pointer);

    static gpu_toolkit& get();
};

template <typename ... Args>
kernel<Args...>::kernel(const std::string& name) : _kernel(gpu_toolkit::get().get_kernel(name)) {}

}}
