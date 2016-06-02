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
#include "ocl_toolkit.h"

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

const std::string kernelCode = R"__krnl(
__kernel void Fully_Connected_GPU(const __global float* input, uint4 input_size, const __global float* weights, uint4 weight_size, __global float* bias, __global float* pDst)
{
    const int x = get_global_id(0);

    pDst[x] = 0;
    uint outXIdx = x / input_size[0];
    uint inputBatchIdx = x % input_size[0];
    uint weightYIdx = outXIdx * weight_size[0];
    for (uint i = 0; i < input_size[2]; i++)
    {
        pDst[x] += input[i * input_size[0] + inputBatchIdx] * weights[weightYIdx + i];
    }
    pDst[x] += bias[outXIdx];
})__krnl";

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

namespace neural {

    struct fully_connected_gpu : is_an_implementation {
        const fully_connected &outer;
        fully_connected_gpu(fully_connected &arg)
            : is_an_implementation(neural::type_id<fully_connected_gpu>())
            , outer(arg)
        {};
        ~fully_connected_gpu() {}

        static void implementation(const void *ptr) {

            auto& toolkit = ocl_toolkit::get();

            auto this_fc = static_cast<const fully_connected *>(ptr);

            // input
            auto& input_mem = this_fc->input_memory(0);
            auto& input_buffer_size = input_mem.argument.size;

            assert(1 == input_buffer_size.feature.size());
            assert(1 == input_buffer_size.batch.size());
            assert(1 == input_buffer_size.feature[0]);

            auto inputSizes = toolkit.get_memory_sizes(input_mem);
            auto inBuffer = toolkit.create_input_buffer<float>(input_mem);

            // weights
            auto& weight_mem = this_fc->input_memory(1);
            auto weightBuffer = toolkit.create_input_buffer<float>(weight_mem);

            auto weightSizes = toolkit.get_memory_sizes(weight_mem);

            // bias
            auto biasBuffer = toolkit.create_input_buffer<float>(this_fc->input_memory(2));

            // output
            auto& output_mem = this_fc->output_memory(0);
            auto& output_arg = output_mem.argument;
            auto& output_buffer_size = output_arg.size;

            auto output_bufSize = std::accumulate(std::begin(output_buffer_size.raw), std::end(output_buffer_size.raw), static_cast<cl::size_type>(1), std::multiplies<cl::size_type>());
            
            auto _outBuffer = toolkit.create_output_buffer<float>(this_fc->output_memory(0));

            auto programKernel = toolkit.getKernel <
                ocl_toolkit::buffer_type,
                cl_uint4,
                ocl_toolkit::buffer_type,
                cl_uint4,
                ocl_toolkit::buffer_type,
                ocl_toolkit::buffer_type
                > ("Fully_Connected_GPU");

            programKernel(
                cl::EnqueueArgs(
                    cl::NDRange(output_bufSize),
                    cl::NDRange(output_bufSize)),
                inBuffer,
                inputSizes,
                weightBuffer,
                weightSizes,
                biasBuffer,
                _outBuffer                
            );

            toolkit.read_buffer<float>(_outBuffer, output_mem);
        }

        task_group work() override {
            return{ { task{ implementation, &outer } }, schedule::single };
        }

        static is_an_implementation *create(fully_connected &arg) { return new fully_connected_gpu(arg); };
    };

namespace {
    struct attach {
        attach() {
            ocl_toolkit::get().add_kernel(kernelCode);
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