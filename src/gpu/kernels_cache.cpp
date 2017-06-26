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
#include <set>

#ifndef NDEBUG
#define OUT_PORGRAM_TO_FILE
#endif

namespace neural { namespace gpu {

const char program_dump_file_name[] = "clDNN_program";

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

#define TRANSPOSE_BLOCK_8_FP16( _block )   \
        (half8)( intel_sub_group_shuffle( _block, 0 ), \
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

#define TRANSPOSE_BLOCK_8_COL_FP16( _block, _col )   \
        (half8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_16_FP16(_block)  \
        (half16)(as_half2(intel_sub_group_shuffle(_block, 0)),  \
                 as_half2(intel_sub_group_shuffle(_block, 1)),  \
                 as_half2(intel_sub_group_shuffle(_block, 2)),  \
                 as_half2(intel_sub_group_shuffle(_block, 3)),  \
                 as_half2(intel_sub_group_shuffle(_block, 4)),  \
                 as_half2(intel_sub_group_shuffle(_block, 5)),  \
                 as_half2(intel_sub_group_shuffle(_block, 6)),  \
                 as_half2(intel_sub_group_shuffle(_block, 7)));

#define TRANSPOSE_BLOCK_16_FP16_HALF_TYPE(_block)  \
        (half16)(intel_sub_group_shuffle(_block, 0),  \
                 intel_sub_group_shuffle(_block, 1),  \
                 intel_sub_group_shuffle(_block, 2),  \
                 intel_sub_group_shuffle(_block, 3),  \
                 intel_sub_group_shuffle(_block, 4),  \
                 intel_sub_group_shuffle(_block, 5),  \
                 intel_sub_group_shuffle(_block, 6),  \
                 intel_sub_group_shuffle(_block, 7),  \
                 intel_sub_group_shuffle(_block, 8),  \
                 intel_sub_group_shuffle(_block, 9),  \
                 intel_sub_group_shuffle(_block, 10),  \
                 intel_sub_group_shuffle(_block, 11),  \
                 intel_sub_group_shuffle(_block, 12),  \
                 intel_sub_group_shuffle(_block, 13),  \
                 intel_sub_group_shuffle(_block, 14),  \
                 intel_sub_group_shuffle(_block, 15));

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

namespace 
{
    std::string get_undef_jit(kernels_cache::source_code org_source_code)
    {
        const std::string white_space_with_new_lines = " \t\r\n";
        const std::string white_space = " \t";

        size_t current_pos = 0;

        const std::string define = "define";

        std::set<std::string> to_undef;
        for (const auto& source : org_source_code)
        {
            do
            {
                size_t index_to_hash = source.find_first_not_of(white_space_with_new_lines, current_pos);
                if (index_to_hash != std::string::npos &&
                    source[index_to_hash] == '#')
                {
                    size_t index_define = source.find_first_not_of(white_space, index_to_hash + 1);

                    if (index_define != std::string::npos &&
                        !source.compare(index_define, define.size(), define))
                    {
                        size_t index_to_name = source.find_first_not_of(white_space, index_define + define.size());
                        if (index_to_name != std::string::npos)
                        {
                            size_t index_to_end_name = source.find_first_of(white_space_with_new_lines + "(", index_to_name);
                            if (index_to_end_name == std::string::npos)
                            {
                                index_to_end_name = source.size();
                            }
                            std::string name = source.substr(index_to_name, index_to_end_name - index_to_name);
                            to_undef.insert(name);
                        }
                        else
                        {
                            //printf("ERROR - %s\n", source.substr(current_pos, 10).c_str());
                        }
                    }
                }
                current_pos = source.find_first_of('\n', current_pos + 1);
            } while (current_pos != std::string::npos);
        }

        std::string undefs;
        for (const auto& name : to_undef)
        {
            undefs += "#ifdef " + name + "\n";
            undefs += "#undef " + name + "\n";
            undefs += "#endif\n";
        }

        return std::move(undefs);
    }

    std::string reoder_options(const std::string& org_options)
    {
        std::stringstream ss(org_options);
        std::set<std::string> sorted_options;
        while (ss.good())
        {
            std::string word;
            ss >> word;
            sorted_options.insert(word);
        }

        std::string options;

        for (const auto& o : sorted_options)
        {
            options += o + " ";
        }
        
        return options;
    }

    inline bool does_options_support_batch_compilation(const std::string& options)
    {
        return
            options.find("-D") == std::string::npos &&
            options.find("-I") == std::string::npos;
    }
}

kernels_cache::sorted_code kernels_cache::get_program_source(const kernels_code& kernels_source_code) const 
{
    sorted_code scode;
    for (auto& code : kernels_source_code)
    {
        const source_code&  org_source_code     = code.second.source;
        std::string         options             = code.second.options;
        bool                batch_compilation   = code.second.batch_compilation;
        bool                inject_header       = code.second.intect_header;

        batch_compilation &= does_options_support_batch_compilation(options);

        if (batch_compilation)
        {
            options = reoder_options(options);
        }

        std::string key = options;

        if (batch_compilation == false)
        {
            key += " __PROGRAM__" + std::to_string(scode.size());
        }
        
        if (inject_header)
        {
            key += " __PROGRAM_INJECT_HEADER__";
        }

        auto& current_bucket = scode[key];

        if (current_bucket.source.empty())
        {
            if (inject_header)
            {
                current_bucket.source.push_back(kernels_header);
            }

            current_bucket.options = options;
        }

        source_code new_source_code = org_source_code;

        if (batch_compilation)
        {
            new_source_code.push_back(get_undef_jit(org_source_code));
        }

        for (auto& s : new_source_code)
        {
            current_bucket.source.push_back(std::move(s));
        }
    }
    return std::move(scode);
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
            std::for_each(std::crbegin(defined_macroses), std::crend(defined_macroses), [&](const std::string& name) { os << "#undef " << name << std::endl; });
            return os.str();
        }
    };

}

kernels_cache::kernels_cache(gpu_toolkit& context): _context(context) {}

kernels_cache::kernel_id kernels_cache::create_kernel_from_template(const std::string& template_name, jit_definitions definitions, std::string kernel_name) 
{
    std::string primitive_name = kernel_name;
    std::replace(kernel_name.begin(), kernel_name.end(), '.', '_');

    if (kernel_name.empty() || !_context.get_configuration().meaningful_kernels_names)
    {
        kernel_name = template_name;
    }

    auto kernel_num = 
        (definitions.empty() && kernel_name == template_name) ? 
        "" : std::to_string(_kernels.size() + _kernels_code.size());

    kernel_name += (kernel_num.empty() ? "" : "_") + kernel_num;
    
    class code_builder code;
    code.add_line("\n//====================================================")
        .add_line("// Kernel template: " + template_name + " ")
        .add_line("// Kernel name: " + kernel_name)
        .add_line("// Primitive id: " + primitive_name)
        .value_macro("KERNEL(name)", "__kernel void " + kernel_name)
        .decoration_macro("FUNC", "", kernel_num)
        .decoration_macro("FUNC_CALL", "", kernel_num);
    
    for (auto& definition : definitions) 
    {
        code.value_macro(definition.first, definition.second);
    }
    code.set_code(_database.get(template_name).at(0));

    auto kernel_code = code.str();

    std::lock_guard<std::mutex> lock(_mutex);
    _kernels_code[kernel_name] = { {kernel_code}, "-cl-mad-enable", true, true };
    return kernel_name;
}

kernels_cache::kernel_id kernels_cache::create_kernel_from_template_ks(const source_code& source, const std::string& options, const std::string& entry_point, bool batch_compilation)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _kernels_code[entry_point] = { source, options, batch_compilation, false };
    return entry_point;
}

kernels_cache::kernels_map kernels_cache::build_program(const program_code& program_source) const
{
#ifdef OUT_PORGRAM_TO_FILE
    static uint32_t current_file_index = 0;
    const std::string current_program_dump_file_name = program_dump_file_name + std::to_string(current_file_index) + ".cl";
    current_file_index++;
#endif
    try {
#ifdef OUT_PORGRAM_TO_FILE
        {
            std::ofstream os(current_program_dump_file_name);
            for (auto& s : program_source.source)
                os << s;
        }
#endif
        cl::Program program(_context.context(), program_source.source);
        program.build({ _context.device() }, program_source.options.c_str());
#ifdef OUT_PORGRAM_TO_FILE
        {
            std::ofstream os(current_program_dump_file_name, std::ios_base::app);
            os << "\n/* Build Log:\n";
            for (auto& p : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
                os << p.second << "\n";
            }
            os << "*/\n";
        }
#endif
        cl::vector<cl::Kernel> kernels;
        program.createKernels(&kernels);
        kernels_map kmap;

        for (auto& k : kernels)
        {
            auto kernel_name = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
            kmap.emplace(kernel_name, k);
        }

        return std::move(kmap);
    }
    catch (const cl::BuildError& err) {
        std::string build_log{"Build program error "};
        build_log += err.what();
#ifdef OUT_PORGRAM_TO_FILE
        {
            std::ofstream os(current_program_dump_file_name, std::ios_base::app);
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

kernels_cache::kernel_type kernels_cache::get_kernel(kernel_id id) 
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (_kernels_code.empty() == false) {

        auto sorted_program_code = get_program_source(_kernels_code);
        _kernels_code.clear();
        for (auto& program : sorted_program_code)
        {
            auto kernels = build_program(program.second);
            _kernels.insert(kernels.begin(), kernels.end());
        }
    }
    return _kernels.at(id);
}

}}
