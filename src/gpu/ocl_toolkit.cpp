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

#define BUFFER_ALIGNMENT 4096
#define CACHE_ALIGNMENT 64

namespace neural { namespace gpu {

const char* definitions_cl = R"__krnl(
enum neural_memory_format {
    x_f32,
    xb_f32,     // 1D+batch, float32
    bx_f32,     // 1D+batch, float32
    yxfb_f32,   // 3D+batch, float32
    byxf_f32,   // for convolution_cpu_jit_batch1
    bfyx_f32,   // used in Caffe
    fyxb_f32,   // used in Caffe
    oiyx_f32,   // format used only for weights: o - output feature maps, i - input feature maps
    byxf_b24_f32,        // for convolution_cpu_generic
    yxoi_o4_f32,       // for convolution_cpu_generic
    os_yxi_sv16_f32,   // format used only for weights: os - output slice, i - input feature maps, sv16 - 16 values of single slice
    bs_yxf_bv24_f32,
    any=-1
};

#pragma pack(push, 4)
typedef struct _neural_memory_tag {
    uint format;
    uint feature_offset;
    uint spatial_offset;
    uint vector_size;
    uint data_offset;
    uint data[1];
} neural_memory;

typedef struct _neural_vector_tag {
    uint feature_offset;
    uint spatial_offset;
    uint raw_size;
    uint data[1];
} neural_vector;
#pragma pack(pop)

// neural_memory accessors
__attribute__((overloadable)) __global uint* get_raw(__global neural_memory* mem) { return &(mem->data[0]); }
__attribute__((overloadable)) const __global uint* get_raw(const __global neural_memory* mem) { return &(mem->data[0]); }
__attribute__((overloadable)) uint get_raw_size(const __global neural_memory* mem) { return mem->vector_size; } 

__attribute__((overloadable)) __global uint* get_batch(__global neural_memory* mem) { return get_raw(mem); }
__attribute__((overloadable)) const __global uint* get_batch(const __global neural_memory* mem) { return get_raw(mem); }
__attribute__((overloadable)) uint get_batch_size(const __global neural_memory* mem) { return mem->feature_offset; }

__attribute__((overloadable)) __global uint* get_feature(__global neural_memory* mem) { return &(mem->data[mem->feature_offset]); }
__attribute__((overloadable)) const __global uint* get_feature(const __global neural_memory* mem) { return &(mem->data[mem->feature_offset]); }
__attribute__((overloadable)) uint get_feature_size(const __global neural_memory* mem) { return mem->spatial_offset - mem->feature_offset; }

__attribute__((overloadable)) __global uint* get_spatial(__global neural_memory* mem) { return &(mem->data[mem->spatial_offset]); }
__attribute__((overloadable)) const __global uint* get_spatial(const __global neural_memory* mem) { return &(mem->data[mem->spatial_offset]); }
__attribute__((overloadable)) uint get_spatial_size(const __global neural_memory* mem) { return get_raw_size(mem) - mem->spatial_offset; } 

__attribute__((overloadable)) __global void* get_data(__global neural_memory* mem) { return &(mem->data[mem->data_offset]); }
__attribute__((overloadable)) const __global void* get_data(const __global neural_memory* mem) { return &(mem->data[mem->data_offset]); }
__attribute__((overloadable)) size_t get_element_size(const __global neural_memory* mem) { return sizeof(float); }

__attribute__((overloadable)) size_t get_data_size(const __global neural_memory* mem) {
    size_t result = get_element_size(mem);

    const __global uint* raw = get_raw(mem);
    uint raw_size = get_raw_size(mem);

    for(uint i = 0; i < raw_size; i++) {
        result *= raw[i];
    }
    return result;
}

// neural_vector accessors
// TODO NOTE: non-const accessors are disabled now, because read-only neural_vector argument is only supported now

//__attribute__((overloadable)) __global uint* get_raw(__global neural_vector* v) { return &(v->data[0]); }
__attribute__((overloadable)) const __global uint* get_raw(const __global neural_vector* v) { return &(v->data[0]); }
__attribute__((overloadable)) uint get_raw_size(const __global neural_vector* v) { return v->raw_size; } 

//__attribute__((overloadable)) __global uint* get_batch(__global neural_vector* v) { return get_raw(v); }
__attribute__((overloadable)) const __global uint* get_batch(const __global neural_vector* v) { return get_raw(v); }
__attribute__((overloadable)) uint get_batch_size(const __global neural_vector* v) { return v->feature_offset; }

//__attribute__((overloadable)) __global uint* get_feature(__global neural_vector* v) { return &(v->data[v->feature_offset]); }
__attribute__((overloadable)) const __global uint* get_feature(const __global neural_vector* v) { return &(v->data[v->feature_offset]); }
__attribute__((overloadable)) uint get_feature_size(const __global neural_vector* v) { return v->spatial_offset - v->feature_offset; }

//__attribute__((overloadable)) __global uint* get_spatial(__global neural_vector* v) { return &(v->data[v->spatial_offset]); }
__attribute__((overloadable)) const __global uint* get_spatial(const __global neural_vector* v) { return &(v->data[v->spatial_offset]); }
__attribute__((overloadable)) uint get_spatial_size(const __global neural_vector* v) { return get_raw_size(v) - v->spatial_offset; } 

)__krnl";

size_t align_to(size_t size, size_t align) {
    return (size % align == 0) ? size : size - size % align + align;
}

size_t pad_to(size_t size, size_t align) {
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
T* arr_begin(T* buf, size_t count) { return buf; }

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
        return std::accumulate( arr_begin(raw_begin(), raw_size()),
                                arr_end(raw_begin(), raw_size()),
                                memory::traits(static_cast<memory::format::type>(format)).type->size,
                                std::multiplies<size_t>()
        );
    }

    void initialize(neural::memory::arguments arg) {
        format = static_cast<cl_uint>(arg.format);
        feature_offset = static_cast<cl_uint>(arg.size.batch.size());
        spatial_offset = static_cast<cl_uint>(arg.size.batch.size() + arg.size.feature.size());
        vector_size = static_cast<cl_uint>(arg.size.raw.size());

        data_offset = static_cast<cl_uint>(get_data_offset(arg));

        std::copy(std::begin(arg.size.raw), std::end(arg.size.raw), arr_begin(raw_begin(), raw_size()));
    }

    static size_t header_size() {
        return sizeof(neural_memory) - sizeof(neural_memory::data);
    }

    static size_t datasize(neural::memory::arguments arg) {
        return std::accumulate( arg.size.raw.begin(),
                                arg.size.raw.end(),
                                memory::traits(arg.format).type->size,
                                std::multiplies<size_t>()
        );
    }

    static size_t get_data_offset(neural::memory::arguments arg) {
        auto header_and_raw_size = header_size() + arg.size.raw.size() * sizeof(cl_uint);
        auto padding = pad_to(header_and_raw_size, CACHE_ALIGNMENT) / sizeof(cl_uint);
        return arg.size.raw.size() + padding;
    }

    static size_t size_of_memory(neural::memory::arguments arg) {
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


std::once_flag gpu_toolkit::ocl_initialized;

vector_arg::vector_arg(const neural::vector<uint32_t>& arg): _vec(arg), _clBuffer(CL_MEM_READ_ONLY, neural_vector::size_of_vector(arg)) {
    auto queue = cl::CommandQueue::getDefault();
    cl::Event end_event;
    auto mapped_vec = reinterpret_cast<neural_vector*>(queue.enqueueMapBuffer(_clBuffer, true, CL_MAP_WRITE, 0, neural_vector::size_of_vector(_vec), 0, &end_event));
    end_event.wait();
    mapped_vec->initialize(_vec);
    queue.enqueueUnmapMemObject(_clBuffer, mapped_vec, 0, &end_event);
    end_event.wait();
}

vector_arg::~vector_arg() {}

memory_arg::memory_arg(const neural::memory& mem, bool copy_input, bool copy_output): _mem(mem), _copy_input(copy_input), _copy_output(copy_output) {
    if (is_own()) {
        _clBuffer = gpu_toolkit::get().unmap_buffer(mem.pointer);
    }
    else {
        auto buf = gpu_toolkit::get().new_memory_buffer(_mem.argument);
        if (_copy_input) {
            auto src = reinterpret_cast<char*>(_mem.pointer);
            auto dst = reinterpret_cast<char*>(buf->pointer());
            auto data_size = buf->data_size();
            std::copy(arr_begin(src, data_size), arr_end(src, data_size), arr_begin(dst, data_size));
        }
        _clBuffer = gpu_toolkit::get().unmap_buffer(buf->pointer());
    }
}

memory_arg::~memory_arg() {
    auto mem_size = neural_memory::size_of_memory(_mem.argument);
    if (is_own()) {
        //TODO remove const_cast: check if .pointer field of gpu owned_memory can be kept unchanged.
        const_cast<neural::memory&>(_mem).pointer = gpu_toolkit::get().map_memory_buffer(_clBuffer, mem_size)->pointer();
    }
    else if (_copy_output) {
        auto buf = gpu_toolkit::get().map_memory_buffer(_clBuffer, mem_size);
        auto src = reinterpret_cast<char*>(buf->pointer());
        auto dst = reinterpret_cast<char*>(_mem.pointer);
        auto data_size = buf->data_size();
        std::copy(arr_begin(src, data_size), arr_end(src, data_size), arr_begin(dst, data_size));
        gpu_toolkit::get().unmap_buffer(buf->pointer());
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

neural_memory* gpu_toolkit::new_memory_buffer(neural::memory::arguments arg) {
    cl::Buffer buffer{ CL_MEM_READ_WRITE, neural_memory::size_of_memory(arg) };
    auto queue = cl::CommandQueue::getDefault();
    cl::Event end_event;
    auto mapped_mem = reinterpret_cast<neural_memory*>(queue.enqueueMapBuffer(buffer, true, CL_MAP_WRITE, 0, neural_memory::size_of_memory(arg), 0, &end_event));
    end_event.wait();
    mapped_mem->initialize(arg);
    auto pointer = mapped_mem->pointer();
    _mapped_memory.push(pointer,{ buffer, mapped_mem });
    return mapped_mem;
}

neural_memory* gpu_toolkit::map_memory_buffer(const cl::Buffer& buf, cl::size_type size, cl_map_flags flags /*= CL_MAP_WRITE*/) {
    auto queue = cl::CommandQueue::getDefault();
    cl::Event end_event;
    auto mapped_mem = reinterpret_cast<neural_memory*>(queue.enqueueMapBuffer(buf, true, flags, 0, size, 0, &end_event));
    end_event.wait();
    auto pointer = mapped_mem->pointer();
    _mapped_memory.push( pointer, {buf, mapped_mem} );
    return mapped_mem;
}

cl::Buffer gpu_toolkit::unmap_buffer(void* pointer) {
    auto queue = cl::CommandQueue::getDefault();
    auto mapped = _mapped_memory.pop(pointer);
    cl::Event end_event;
    queue.enqueueUnmapMemObject(mapped.first, mapped.second, 0, &end_event);
    end_event.wait();
    return mapped.first;
}

void* allocate_memory_gpu(neural::memory::arguments arg) {
    auto mapped_mem = gpu_toolkit::get().new_memory_buffer(arg);
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
