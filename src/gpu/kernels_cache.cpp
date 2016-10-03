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
#include "kernels_cache.h"
#include "ocl_toolkit.h"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <fstream>

namespace neural { namespace gpu {

const char program_dump_file_name[] = "clDNN_program.cl";

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

#define neural_memory float

__attribute__((overloadable)) __global void* get_data(__global neural_memory* mem) { return mem; }
__attribute__((overloadable)) const __global void* get_data(const __global neural_memory* mem) { return mem; }

#define TRANSPOSE_BLOCK_8( _block )   \
        (float8)( intel_sub_group_shuffle( _block, 0 ), \
                  intel_sub_group_shuffle( _block, 1 ), \
                  intel_sub_group_shuffle( _block, 2 ), \
                  intel_sub_group_shuffle( _block, 3 ), \
                  intel_sub_group_shuffle( _block, 4 ), \
                  intel_sub_group_shuffle( _block, 5 ), \
                  intel_sub_group_shuffle( _block, 6 ), \
                  intel_sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_COL( _block, _col )   \
        (float8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB)  \
{   \
    const float8 acol0 = TRANSPOSE_BLOCK_8_COL( _blockA, 0 ); \
    const float8 acol1 = TRANSPOSE_BLOCK_8_COL( _blockA, 1 ); \
    const float8 acol2 = TRANSPOSE_BLOCK_8_COL( _blockA, 2 ); \
    const float8 acol3 = TRANSPOSE_BLOCK_8_COL( _blockA, 3 ); \
    const float8 acol4 = TRANSPOSE_BLOCK_8_COL( _blockA, 4 ); \
    const float8 acol5 = TRANSPOSE_BLOCK_8_COL( _blockA, 5 ); \
    const float8 acol6 = TRANSPOSE_BLOCK_8_COL( _blockA, 6 ); \
    const float8 acol7 = TRANSPOSE_BLOCK_8_COL( _blockA, 7 ); \
    _result = mad( _blockB.s0, acol0, _result ); \
    _result = mad( _blockB.s1, acol1, _result ); \
    _result = mad( _blockB.s2, acol2, _result ); \
    _result = mad( _blockB.s3, acol3, _result ); \
    _result = mad( _blockB.s4, acol4, _result ); \
    _result = mad( _blockB.s5, acol5, _result ); \
    _result = mad( _blockB.s6, acol6, _result ); \
    _result = mad( _blockB.s7, acol7, _result ); \
}

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
{   \
        _result.s0 = mad( _rowA, intel_sub_group_shuffle( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, intel_sub_group_shuffle( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, intel_sub_group_shuffle( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, intel_sub_group_shuffle( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, intel_sub_group_shuffle( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, intel_sub_group_shuffle( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, intel_sub_group_shuffle( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, intel_sub_group_shuffle( colB, 7 ), _result.s7 );  \
}
#define ADD_BIAS_8( _result, _biasVal) \
{ \
    _result.s0 += intel_sub_group_shuffle( _biasVal, 0 ); \
    _result.s1 += intel_sub_group_shuffle( _biasVal, 1 ); \
    _result.s2 += intel_sub_group_shuffle( _biasVal, 2 ); \
    _result.s3 += intel_sub_group_shuffle( _biasVal, 3 ); \
    _result.s4 += intel_sub_group_shuffle( _biasVal, 4 ); \
    _result.s5 += intel_sub_group_shuffle( _biasVal, 5 ); \
    _result.s6 += intel_sub_group_shuffle( _biasVal, 6 ); \
    _result.s7 += intel_sub_group_shuffle( _biasVal, 7 ); \
}

#define ACTIVATION_8(_result) \
{ \
        ACTIVATION(_result.s0, _result.s0); \
        ACTIVATION(_result.s1, _result.s1); \
        ACTIVATION(_result.s2, _result.s2); \
        ACTIVATION(_result.s3, _result.s3); \
        ACTIVATION(_result.s4, _result.s4); \
        ACTIVATION(_result.s5, _result.s5); \
        ACTIVATION(_result.s6, _result.s6); \
        ACTIVATION(_result.s7, _result.s7); \
}

)__krnl";

std::vector<std::string> kernels_cache::get_program_source() const {
    std::vector<std::string> source{ kernels_header };
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

        code_builder& add_line(const std::string& line) {
            oss << line << "\n";
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

kernels_cache::kernel_id kernels_cache::create_kernel_from_template(const std::string& template_name, jit_definitions definitions) {
    // TODO: FIXIT: more than one kernel can be created for same template_name and definitions
    auto kernel_num = definitions.empty() ? "" : std::to_string(_kernel_codes.size());
    auto kernel_name = template_name + (kernel_num.empty() ? "" : "_") + kernel_num;

    class code_builder code;
    code.add_line("\n//====================================================")
        .add_line("// Kernel template: " + template_name + " ")
        .add_line("// Kernel name: " + kernel_name)
        .decoration_macro("KERNEL", "__kernel void", kernel_num)
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

void kernels_cache::build_program() {
    try {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_modified) {
            auto program_source = get_program_source();
#ifndef NDEBUG
            {
                std::ofstream os(program_dump_file_name);
                for (auto& s : program_source)
                    os << s;
            }
#endif
            cl::Program program(_context.context(), program_source);
            program.build({ _context.device() }, "-cl-mad-enable");
#ifndef NDEBUG
            {
                std::ofstream os(program_dump_file_name, std::ios_base::app);
                os << "\n/* Build Log:\n";
                for (auto& p : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
                    os << p.second << "\n";
                }
                os << "*/\n";
            }
#endif
            cl::vector<cl::Kernel> kernels;
            program.createKernels(&kernels);
            _kernels.clear();
            for(auto& k : kernels)
            {
                auto kernel_name = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                _kernels.emplace(kernel_name, k);
            }
        }
        _modified = false;
    }
    catch (const cl::BuildError& err) {
        std::string build_log{"Build program error "};
        build_log += err.what();
#ifndef NDEBUG
        {
            std::ofstream os(program_dump_file_name, std::ios_base::app);
            os << "\n/* Build Log:\n";
            for (auto& p : err.getBuildLog()) {
                os << p.second << "\n";
                build_log += "\n" + p.second;
            }
            os << "*/\n";
        }
#endif
        throw std::runtime_error(build_log);
    }
}

kernels_cache::kernel_type kernels_cache::get_kernel(kernel_id id) {
    build_program();
    return _kernels.at(id);
}

}}
