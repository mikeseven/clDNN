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

#include "igk_kernel_base.h"

#if defined __INTEL_COMPILER
#pragma warning disable: 177
#endif

namespace KernelSelector 
{
static const char* kernels_header = R"__krnl(
#define CAT(x, y) x##y
#define LOOP1(VAR, STMT) (STMT); (VAR)++;
#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;
#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;
#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;
#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;
#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;
#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;
#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;
#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;
#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;
#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;
#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;
#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;
#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))

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

#define TRANSPOSE_BLOCK_16_FP16(_block)                                  \
        (half16)(as_half2(intel_sub_group_shuffle(_block, 0)),  \
                 as_half2(intel_sub_group_shuffle(_block, 1)),  \
                 as_half2(intel_sub_group_shuffle(_block, 2)),  \
                 as_half2(intel_sub_group_shuffle(_block, 3)),  \
                 as_half2(intel_sub_group_shuffle(_block, 4)),  \
                 as_half2(intel_sub_group_shuffle(_block, 5)),  \
                 as_half2(intel_sub_group_shuffle(_block, 6)),  \
                 as_half2(intel_sub_group_shuffle(_block, 7)));

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

#define ADD_BIAS_16_FP16( _result, _biasVal) \
{ \
    _result.s01 += as_half2(intel_sub_group_shuffle(_biasVal, 0)); \
    _result.s23 += as_half2(intel_sub_group_shuffle(_biasVal, 1)); \
    _result.s45 += as_half2(intel_sub_group_shuffle(_biasVal, 2)); \
    _result.s67 += as_half2(intel_sub_group_shuffle(_biasVal, 3)); \
    _result.s89 += as_half2(intel_sub_group_shuffle(_biasVal, 4)); \
    _result.sab += as_half2(intel_sub_group_shuffle(_biasVal, 5)); \
    _result.scd += as_half2(intel_sub_group_shuffle(_biasVal, 6)); \
    _result.sef += as_half2(intel_sub_group_shuffle(_biasVal, 7)); \
}

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + byte_offset))

#define MULTIPLY_OFFSET(elem_type, byte_offset) (byte_offset * sizeof(elem_type))

)__krnl";

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

            code_builder& decoration_macro(const std::string& name, const std::string& prefix, const std::string& postfix, const std::string& name_prefix = std::string())
            {
                oss << "#define " << name << "(name) " << prefix << " " + name_prefix + "_##" + "name" << (postfix.empty() ? "" : "##_") << postfix << std::endl;
                return register_macro(name);
            }


            code_builder& value_macro(const std::string& name, const std::string& value)
            {
                oss << "#define " << name << " " << value << std::endl;
                return register_macro(name.substr(0, name.find('(')));
            }

            std::string str()
            {
                std::ostringstream os;
                os << oss.str();
                os << code << std::endl;
                return os.str();
            }
        };
    }

    std::string IGKKernelBase::create_jit_from_template(const std::string& template_name, jit_definitions definitions, std::string kernel_id) const
    {
        std::replace(kernel_id.begin(), kernel_id.end(), '.', '_');

        if (kernel_id.empty() /*|| !_context.get_configuration().meaningful_kernels_names*/)
        {
            kernel_id = template_name;
        }

        class code_builder code;
        code.add_line("\n//====================================================")
            .add_line("// Kernel template: " + template_name + " ")
            .add_line("// Kernel name: " + kernel_id)
            .value_macro("KERNEL(name)", "__kernel void " + kernel_id)
            .decoration_macro("FUNC", "", kernel_id)
            .decoration_macro("FUNC_CALL", "", kernel_id);
        
        for (auto& definition : definitions) 
        {
            code.value_macro(definition.first, definition.second);
        }

        auto jit = std::string(kernels_header) + code.str();

        return jit;
    }

    ArgumentDescpirtor IGKKernelBase::get_args_desc(uint32_t num_of_input, bool use_weights, bool use_bias) const
    {
        ArgumentDescpirtor desc;

        for (uint32_t i = 0; i < num_of_input; i++)
        {
            desc.data.push_back({ ArgumentDescpirtor::Types::INPUT, 0 });
        }

        desc.data.push_back({ ArgumentDescpirtor::Types::OUTPUT, 0 });

        if (use_weights)
        {
            desc.data.push_back({ ArgumentDescpirtor::Types::WEIGHTS, 0 });
        }

        if (use_bias)
        {
            desc.data.push_back({ ArgumentDescpirtor::Types::BIAS, 0 });
        }

        return desc;
    }

    KernelString IGKKernelBase::get_kernel_string(std::string name, std::string jit, std::string entry_point, std::string exe_mode) const
    {
        KernelString kernel_string;

        auto codes = db.get(name);

        if (codes.size())
        {
            kernel_string.str = codes[0];
            kernel_string.jit = jit;
            kernel_string.options = exe_mode + " -cl-mad-enable";
            kernel_string.entry_point = entry_point;
            kernel_string.batch_compilation = true;
        }

        return kernel_string;
    }
}