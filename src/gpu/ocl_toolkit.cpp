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
#include "ocl_toolkit.h"
#include "memory.h"

namespace neural { namespace gpu {

const char* definitions_cl = R"__krnl(
#pragma pack(push, 1)
typedef struct _neural_memory_tag {
    uint format;
    uint feature_offset;
    uint spatial_offset;
    uint data_offset;
    uint data[1];
} neural_memory;
#pragma pack(pop)

__global uint* get_raw(__global neural_memory* mem) { return &(mem->data[0]); }
uint get_raw_size(__global neural_memory* mem) { return mem->data_offset; } 

__global uint* get_batch(__global neural_memory* mem) { return get_raw(mem); }
uint get_batch_size(__global neural_memory* mem) { return mem->feature_offset; }

__global uint* get_feature(__global neural_memory* mem) { return &(mem->data[mem->feature_offset]); }
uint get_feature_size(__global neural_memory* mem) { return mem->spatial_offset - mem->feature_offset; }

__global uint* get_spatial(__global neural_memory* mem) { return &(mem->data[mem->spatial_offset]); }
uint get_spatial_size(__global neural_memory* mem) { return mem->data_offset - mem->spatial_offset; } 

__global void* get_data(__global neural_memory* mem) { return &(mem->data[mem->data_offset]); }
size_t get_element_size(__global neural_memory* mem) { return sizeof(float); }

size_t get_data_size(__global neural_memory* mem) {
    size_t result = get_element_size(mem);

    __global uint* raw = get_raw(mem);
    uint raw_size = get_raw_size(mem);

    for(uint i = 0; i < raw_size; i++) {
        result *= raw[i];
    }
    return result;
}
)__krnl";

#pragma pack(push, 1)
struct neural_memory {
    cl_uint format;
    cl_uint feature_offset;
    cl_uint spatial_offset;
    cl_uint data_offset;
    cl_uint data[1];

    void* pointer() { return reinterpret_cast<void*>(raw_end()); }
    cl_uint* raw_begin() { return data; }
    cl_uint* raw_end() { return data + data_offset; }
    size_t data_size() {
        auto result = memory::traits(static_cast<memory::format::type>(format)).type->size;
        for (cl_uint i = 0; i < raw_end() - raw_begin(); i++)
            result *= data[i];
        return result;
    }

    void initialize(neural::memory::arguments arg) {
        format = static_cast<cl_uint>(arg.format);
        feature_offset = static_cast<cl_uint>(arg.size.batch.size());
        spatial_offset = static_cast<cl_uint>(arg.size.batch.size() + arg.size.feature.size());
        data_offset = static_cast<cl_uint>(arg.size.raw.size());

#if defined(_MSC_VER)
        auto dst = stdext::make_checked_array_iterator(raw_begin(), data_offset);
#else
        auto dst = raw_begin();
#endif
        std::copy(std::begin(arg.size.raw), std::end(arg.size.raw), dst);
    }

    static size_t header_size() {
        return sizeof(neural_memory) - sizeof(neural_memory::data);
    }

    static size_t datasize(neural::memory::arguments arg) {
        auto count = std::accumulate(arg.size.raw.begin(), arg.size.raw.end(), size_t(1), std::multiplies<size_t>());
        auto elem_size = memory::traits(arg.format).type->size;
        return count * elem_size;
    }

    static size_t size(neural::memory::arguments arg) {
        return header_size() + arg.size.raw.size() * sizeof(cl_uint) + datasize(arg);
    }

    size_t size() {
        return  header_size() + (raw_end() - raw_begin()) * sizeof(cl_uint) + data_size();
    }
};
#pragma pack(pop)

std::once_flag gpu_toolkit::ocl_initialized;

cl::size_type buffer::size() const {
    return neural_memory::datasize(_mem.argument);
}

buffer::buffer(const neural::memory& mem, bool copy_input, bool copy_output): _mem(mem), _copy_input(copy_input), _copy_output(copy_output) {
    if (is_own()) {
        _clBuffer = gpu_toolkit::get().unmap_buffer(mem.pointer);
    }
    else if (_copy_input) {
        _clBuffer = cl::Buffer(CL_MEM_COPY_HOST_PTR, size(), _mem.pointer);
    }
}

buffer::~buffer() {
    if (is_own()) {
        //TODO remove const_cast: check if .pointer field of gpu owned_memory can be kept unchanged.
        const_cast<neural::memory&>(_mem).pointer = gpu_toolkit::get().map_buffer(_clBuffer, size())->pointer();
    }
    else if (_copy_output) {
        auto output = static_cast<char*>(_mem.pointer);
#if defined(_MSC_VER)
        auto out_begin = stdext::make_checked_array_iterator(output, size());
        auto out_end = stdext::make_checked_array_iterator(output, size(), size());
#else
        auto out_begin = output;
        auto out_end = output + size();
#endif
        cl::copy(_clBuffer, out_begin, out_end);
    }
}

gpu_toolkit::gpu_toolkit() {
    std::call_once(ocl_initialized, initialize);
    add_kernel(definitions_cl);
}

void gpu_toolkit::initialize() {
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

neural_memory* gpu_toolkit::new_buffer(neural::memory::arguments arg) {
    cl::Buffer buffer(CL_MEM_READ_WRITE, neural_memory::size(arg));
    auto queue = cl::CommandQueue::getDefault();
    cl::Event end_event;
    auto mapped_mem = reinterpret_cast<neural_memory*>(queue.enqueueMapBuffer(buffer, true, CL_MAP_WRITE, 0, neural_memory::size(arg), 0, &end_event));
    end_event.wait();
    mapped_mem->initialize(arg);
    auto pointer = mapped_mem->pointer();
    _mapped_buffers.insert({ pointer,{ buffer, mapped_mem } });
    return mapped_mem;
}

neural_memory* gpu_toolkit::map_buffer(const cl::Buffer& buf, cl::size_type size, cl_map_flags flags /*= CL_MAP_WRITE*/) {
    auto queue = cl::CommandQueue::getDefault();
    cl::Event end_event;
    auto mapped_mem = reinterpret_cast<neural_memory*>(queue.enqueueMapBuffer(buf, true, flags, 0, size, 0, &end_event));
    end_event.wait();
    auto pointer = mapped_mem->pointer();
    _mapped_buffers.insert({ pointer, {buf, mapped_mem} });
    return mapped_mem;
}

cl::Buffer gpu_toolkit::unmap_buffer(void* pointer) {
    auto queue = cl::CommandQueue::getDefault();
    auto it = _mapped_buffers.find(pointer);
    if (it == std::end(_mapped_buffers)) throw std::runtime_error("The pointer is not the mapped buffer");
    auto mapped = it->second;
    _mapped_buffers.erase(it);
    cl::Event end_event;
    queue.enqueueUnmapMemObject(mapped.first, mapped.second, 0, &end_event);
    end_event.wait();
    return mapped.first;
}

void* allocate_memory_gpu(neural::memory::arguments arg) {
    auto mapped_mem = gpu_toolkit::get().new_buffer(arg);
    return mapped_mem->pointer();
}

void deallocate_memory_gpu(void* pointer, neural::memory::arguments) {
    gpu_toolkit::get().unmap_buffer(pointer);
}

gpu_toolkit& gpu_toolkit::get() {
    static gpu_toolkit toolkit;
    return toolkit;
}

namespace {
    struct attach_gpu_allocator {
        attach_gpu_allocator() {
            allocators_map::instance().insert({ engine::gpu, {allocate_memory_gpu, deallocate_memory_gpu} });
        }
        ~attach_gpu_allocator() {}
    };

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach_gpu_allocator attach_impl;
}

}}
