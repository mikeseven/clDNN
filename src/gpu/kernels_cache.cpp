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
#include "kernels_cache.h"

namespace neural { namespace gpu {

static const char* kernels_header = R"__krnl(
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

cl::Program::Sources kernels_cache::get_program_source() const {
    cl::Program::Sources source{ kernels_header };
    for (auto& code : _kernel_codes) {
        source.push_back(code.second);
    }
    return source;
}

namespace {

    class code_builder
    {
        std::ostringstream oss;
        std::string code;
        std::vector<std::string> defined_macroses;

        code_builder& register_macro(const std::string& name)
        {
            assert(std::count(defined_macroses.begin(), defined_macroses.end(), name) == 0);
            defined_macroses.push_back(name);
            return *this;
        }

    public:
        code_builder& set_code(const std::string& c)
        {
            assert(code.empty());
            code = c;
            return *this;
        }

        code_builder& decoration_macro(const std::string& name, const std::string& prefix, const std::string& postfix)
        {
            oss << "#define " << name << "(name) " << prefix << " name" << (postfix.empty() ? "" : "##_") << postfix << std::endl;
            return register_macro(name);
        }

        code_builder& value_macro(const std::string& name, const std::string& value)
        {
            oss << "#define " << name << " " << value << std::endl;
            return register_macro(name);
        }

        std::string str()
        {
            std::ostringstream os;
            os << oss.str();
            os << code << std::endl;
            std::for_each(std::crbegin(defined_macroses), std::crend(defined_macroses), [&](const std::string& name) { os << "#undef " << name << std::endl; });
            return os.str();
        }
    };

}

kernels_cache::kernel_id kernels_cache::create_kernel_from_template(std::shared_ptr<neural::gpu::gpu_toolkit>, const std::string& template_name, jit_definitions definitions) {
    // TODO: FIXIT: more than one kernel can be created for same template_name and definitions
    auto kernel_num = definitions.empty() ? "" : std::to_string(_kernel_codes.size());
    auto kernel_name = template_name + (kernel_num.empty() ? "" : "_") + kernel_num;

    class code_builder code;
    code.decoration_macro("KERNEL", "__kernel void", kernel_num)
        .decoration_macro("FUNC", "", kernel_num)
        .decoration_macro("FUNC_CALL", "", kernel_num);
    for (auto& definition : definitions) {
        code.value_macro(definition.first, definition.second);
    }

    code.set_code(kernel_templates::get(template_name));

    auto kernel_code = code.str();

    std::lock_guard<std::mutex> lock(_mutex);
    _kernel_codes[kernel_name] = kernel_code;
    _modified = true;
    return kernel_name;
}

cl::Program kernels_cache::get_program(std::shared_ptr<neural::gpu::gpu_toolkit> context) {
    assert(context != nullptr);
    std::lock_guard<std::mutex> lock(_mutex);
    if (_modified){
        context->program() = cl::Program(context->context(), get_program_source());
        context->program().build({ context->device() });
    }
    _modified = false;
    return context->program();
}

cl::Kernel kernels_cache::get_kernel(std::shared_ptr<neural::gpu::gpu_toolkit> context, kernel_id id) {
    assert(context != nullptr);
    return cl::Kernel(get_program(context), id.c_str());
}

kernels_cache& kernels_cache::get() {
    static kernels_cache instance;
    return instance;
}

}}