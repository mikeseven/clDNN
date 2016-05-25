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
#include "api/neural.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"
#include "fully_connected.h"

#pragma warning (push)
#pragma warning(disable : 4100)
#pragma warning(disable : 4505)
// we want exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>
#pragma warning (pop)

#include <iostream>
#include <fstream>

// wrapper to enable iterators on raw pointers

/*template<typename T>
struct PtrWrapper
{
    class iterator
    {
        public:
            typedef iterator self_type;
            typedef T value_type;
            typedef T& reference;
            typedef T* pointer;
            typedef std::forward_iterator_tag iterator_category;
            typedef int difference_type;
            iterator(pointer ptr) : ptr_(ptr) { }
            self_type operator++() { self_type i = *this; ptr_++; return i; }
            self_type operator++(int junk) { ptr_++; return *this; }
            size_t operator-(self_type it) { return (it.ptr_ - this->ptr_) / sizeof(value_type); }
            reference operator*() { return *ptr_; }
            pointer operator->() { return ptr_; }
            bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
            bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
        private:
            pointer ptr_;
    };

    iterator begin()
    {
        return iterator(b);
    }

    iterator end()
    {
        return iterator(b + size);
    }

    PtrWrapper(T *begin, size_t sz)
        :b(begin), size(sz) {}
    T *b;
    size_t size;
};
template<typename T>
T* begin(PtrWrapper<T> ptr) { return ptr.begin(); }
template<typename T>
T* end(PtrWrapper<T> ptr) { return ptr.end(); }*/

const std::string kernelCode =
"__kernel void Fully_Connected_GPU(const __global float* input, uint input_size, const __global float* weights, uint4 weight_size, __global float* bias, __global float* pDst)\n"
"{\n"
"    const int x = get_global_id(0);\n"
"\n"
"    pDst[x] = 0;\n"
"    for (uint i = 0; i < input_size; i++)\n"
"    {\n"
"        pDst[x] += input[i] * weights[(x * weight_size.x) + i];\n"
"    }\n"
"    pDst[x] += bias[x];\n"
"};\n";

// simple ocl implementation
void initOCLDevice()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform plat;
    for (auto &p : platforms)
    {
        std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
        if (platver.find("OpenCL 2.") != std::string::npos)
        {
            plat = p;
        }
    }
    if (plat() == 0)
    {
        throw std::runtime_error("No OpenCL 2.0 platform found.");
    }

    cl::Platform newP = cl::Platform::setDefault(plat);
    if (newP != plat) 
    {
        throw std::runtime_error("Error setting default platform.");
    }
}

void loadFile(const char *filePath, std::vector<char> &outFile)
{
    std::streampos size;
    std::ifstream file;
    file.open(filePath, std::ios::ate);
    if (file.is_open())
    {
        size = file.tellg();
        outFile.resize(size);
        file.seekg(0, std::ios::beg);
        file.read(&outFile[0], size);
        file.close();
    }
    else
        throw std::runtime_error("Cannot open kernel file.");
}

void compileKernel(cl::Program &outProgram)
{
    try {
        outProgram.build("-cl-std=CL2.0");
    }
    catch (...) {
        // Print build info for all devices
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = outProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) 
        {
            std::cerr << pair.second << std::endl << std::endl;
        }
    }
}

template<typename T>
cl::Buffer CreateBufferWithGoodSize(const neural::memory &mem)
{
    auto data = static_cast<T*>(mem.pointer);
    auto& data_arg = mem.argument;
    auto& data_size = data_arg.size;

    cl_int err;
    cl::size_type data_bufSize = 1;
    // 
    for (auto i : data_size.raw)
    {
        data_bufSize *= i;
    }

    return cl::Buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * data_bufSize, data, &err);
}

cl_uint4 GetSizes(const neural::vector<uint32_t> size)
{
    if (size.raw.size() < 4)
    {
        cl_uint4 x = { 0 };
        for (int i = 0; i < size.raw.size(); i++)
        {
            x.s[i] = size.raw[i];
        }
        return x;
    }
    else
    {
        throw std::runtime_error("Want to get size, but this is not a vector of sizes, because size is greater than 4!");
    }
}

namespace neural {

    struct fully_connected_gpu : is_an_implementation {
        const fully_connected &outer;
        fully_connected_gpu(fully_connected &arg)
            : is_an_implementation(neural::type_id<fully_connected_gpu>())
            , outer(arg)
        {};
        ~fully_connected_gpu() {}

        static void implementation(const void *ptr) {
            initOCLDevice();
            //std::vector<char> kernelFile;
            //loadFile("fully_connected_gpu.cl", kernelFile);
            cl::Program kernel(kernelCode);//&kernelFile[0]);
            compileKernel(kernel);

            auto this_fc = static_cast<const fully_connected *>(ptr);

            // input
            auto& input_arg = this_fc->input_memory(0).argument;
            auto& input_buffer_size = input_arg.size;

            assert(1 == input_buffer_size.feature.size());
            assert(1 == input_buffer_size.batch.size());
            assert(1 == input_buffer_size.feature[0]);

            uint32_t input_bufSize = 1;
            for (auto i : input_buffer_size.raw)
            {
                input_bufSize *= i;
            }
            cl::Buffer inBuffer = CreateBufferWithGoodSize<float>(this_fc->input_memory(0));

            // weights
            cl::Buffer weightBuffer = CreateBufferWithGoodSize<float>(this_fc->input_memory(1));

            auto& weight_arg = this_fc->input_memory(1).argument;
            cl_uint4 weightSizes = GetSizes(weight_arg.size);

            // bias
            cl::Buffer biasBuffer = CreateBufferWithGoodSize<float>(this_fc->input_memory(2));

            // output
            auto output = static_cast<float*>(this_fc->output_memory(0).pointer);
            auto& output_arg = this_fc->output_memory(0).argument;
            auto& output_buffer_size = output_arg.size;

            cl::size_type output_bufSize = 1;
            for (auto i : output_buffer_size.raw)
            {
                output_bufSize *= i;
            }
            
            std::vector<float> _out(output_bufSize, 0.0f);

            cl::Buffer _outBuffer = CreateBufferWithGoodSize<float>(this_fc->output_memory(0));

            cl::DeviceCommandQueue cmdQueue = cl::DeviceCommandQueue::makeDefault(cl::Context::getDefault(), cl::Device::getDefault());

            
            auto programKernel = cl::KernelFunctor <
                cl::Buffer,
                uint32_t,
                cl::Buffer,
                cl_uint4,
                cl::Buffer,
                cl::Buffer
                >(kernel, "Fully_Connected_GPU");

            programKernel(
                cl::EnqueueArgs(
                    cl::NDRange(output_bufSize),
                    cl::NDRange(output_bufSize)),
                inBuffer,
                input_bufSize,
                weightBuffer,
                weightSizes,
                biasBuffer,
                _outBuffer                
            );

            // TODO: get rid of this copy, use custom iterators to enable cl::copy to output directly to "output".
            cl::copy(_outBuffer, begin(_out), end(_out));
            for (int i = 0; i < output_bufSize; i++)
            {
                output[i] = _out[i];
            }
        }

        std::vector<task> work() {
            return{ task{ implementation, &outer } };
        }

        static is_an_implementation *create(fully_connected &arg) { return new fully_connected_gpu(arg); };
    };

namespace {
    struct attach {
        attach() {
            auto val_fw = fully_connected_gpu::create;
            fully_connected_fw_implementation_map::instance().insert({ std::make_tuple(engine::gpu, memory::format::xb_f32, memory::format::xb_f32), val_fw });
            fully_connected_fw_implementation_map::instance().insert({ std::make_tuple(engine::gpu, memory::format::x_f32,  memory::format::x_f32), val_fw });
        }
        ~attach() {}
    };

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}
}