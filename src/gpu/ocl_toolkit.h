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

namespace neural {

class ocl_toolkit {
public:
    using buffer_type = cl::Buffer;

private:
    bool program_modified = true;
    cl::Program::Sources kernel_sources;

    std::unique_ptr<cl::Program> program;

    ocl_toolkit() {
        std::call_once(ocl_initialized, initialize_opencl);
    }

    static std::once_flag ocl_initialized;
    static void initialize_opencl() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform plat;
        for (auto& p : platforms) {
            std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
            if (platver.find("OpenCL 2.") != std::string::npos) {
                plat = p;
            }
        }

        if (plat() == nullptr) {
            throw std::runtime_error("No OpenCL 2.0 platform found.");
        }

        cl::Platform newP = cl::Platform::setDefault(plat);
        if (newP != plat) {
            throw std::runtime_error("Error setting default platform.");
        }
    }

    const cl::Program& get_program() {
        if (!program || program_modified) {
            program.reset(new cl::Program(kernel_sources));
            program->build("-cl-std=CL2.0");
            program_modified = false;
        }
        return *program.get();
    }

public:
    using buffer_type = cl::Buffer;

    void add_kernel(const std::string& source) {
        kernel_sources.push_back(source);
        program_modified = true;
    }

    template<typename... Ts>
    cl::KernelFunctor<Ts...> getKernel(const std::string& name) {
        return cl::KernelFunctor<Ts...>(get_program(), name);
    }

    template<typename T>
    static buffer_type create_input_buffer(const neural::memory& mem) {
        auto data_bufSize = get_buffer_size(mem);

        return cl::Buffer(CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * data_bufSize, mem.pointer);
    }

    template<typename T>
    static buffer_type create_output_buffer(const neural::memory& mem) {
        auto data_bufSize = get_buffer_size(mem);

        return cl::Buffer(CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(T) * data_bufSize);
    }

    template<typename T>
    static buffer_type create_inout_buffer(const neural::memory& mem) {
        auto data_bufSize = get_buffer_size(mem);

        return cl::Buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(T) * data_bufSize, mem.pointer);
    }

    template<typename T>
    static cl_int read_buffer(const buffer_type& buffer, const neural::memory& mem) {
        auto output = static_cast<T*>(mem.pointer);

        auto out_buf_size = get_buffer_size(mem);

#if defined(_MSC_VER)
        auto out_begin = stdext::make_checked_array_iterator(output, out_buf_size);
        auto out_end = stdext::make_checked_array_iterator(output, out_buf_size, out_buf_size);
#else
        auto out_begin = output;
        auto out_end = output + data_bufSize;
#endif
        return cl::copy(buffer, out_begin, out_end);
    }

    static cl::size_type get_buffer_size(const neural::memory& mem) {
        auto& sizes = mem.argument.size.raw;

        return std::accumulate(std::begin(sizes), std::end(sizes), static_cast<cl::size_type>(1), std::multiplies<cl::size_type>{});
    }

    static cl_uint4 get_memory_sizes(const neural::memory& mem)
    {
        auto size = mem.argument.size;
        if (size.raw.size() > 4) {
            throw std::runtime_error("Want to get size, but this is not a vector of sizes, because size is greater than 4!");
        }

        cl_uint4 x = { 0 };
        std::copy(std::begin(size.raw), std::end(size.raw), x.s);
        return x;
    }

    static ocl_toolkit& get() {
        static ocl_toolkit toolkit;
        return toolkit;
    }
};
}
