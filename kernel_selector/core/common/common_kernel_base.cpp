﻿/*
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

#include "common_kernel_base.h"

#if defined __INTEL_COMPILER
#pragma warning disable: 177
#endif

namespace KernelSelector 
{
    namespace {

        class CodeBuilder
        {
            std::ostringstream oss;
            std::string code;
            std::vector<std::string> defined_macroses;

            CodeBuilder& register_macro(const std::string& name)
            {
                assert(std::count(defined_macroses.begin(), defined_macroses.end(), name) == 0);
                defined_macroses.push_back(name);
                return *this;
            }

        public:
            CodeBuilder& set_code(const std::string& c)
            {
                assert(code.empty());
                code = c;
                return *this;
            }

            CodeBuilder& add_line(const std::string& line) {
                oss << line << "\n";
                return *this;
            }

            CodeBuilder& decoration_macro(const std::string& name, const std::string& prefix, const std::string& postfix, const std::string& name_prefix = std::string())
            {
                oss << "#define " << name << "(name) " << prefix << " " + name_prefix + "_##" + "name" << (postfix.empty() ? "" : "##_") << postfix << std::endl;
                return register_macro(name);
            }


            CodeBuilder& value_macro(const std::string& name, const std::string& value)
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

    std::string CommonKernelBase::GetEntryPoint(const std::string& template_name, const std::string& layer_id) const
    {
        std::string kernel_id = layer_id;

        std::replace(kernel_id.begin(), kernel_id.end(), '.', '_');

        if (kernel_id.empty() /*|| !_context.get_configuration().meaningful_kernels_names*/)
        {
            kernel_id = template_name;
        }

        kernel_id += std::to_string(UniqeID());

        return kernel_id;
    }

    std::string CommonKernelBase::CreateJit(const std::string& template_name, JitConstants constants, std::string kernel_id) const
    {
        class CodeBuilder code;
        code.add_line("\n//====================================================")
            .add_line("// Kernel template: " + template_name + " ")
            .add_line("// Kernel name: " + kernel_id)
            .value_macro("KERNEL(name)", "__kernel void " + kernel_id)
            .decoration_macro("FUNC", "", kernel_id)
            .decoration_macro("FUNC_CALL", "", kernel_id);
        
        for (auto& definition : constants.GetDefinitions())
        {
            code.value_macro(definition.first, definition.second);
        }

        std::string jit = code.str();

        return jit;
    }

    ArgumentDescriptor CommonKernelBase::GetArgsDesc(uint32_t num_of_input, bool use_weights, bool use_bias) const
    {
        ArgumentDescriptor desc;

        for (uint32_t i = 0; i < num_of_input; i++)
        {
            desc.data.push_back({ ArgumentDescriptor::Types::INPUT, 0 });
        }

        desc.data.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });

        if (use_weights)
        {
            desc.data.push_back({ ArgumentDescriptor::Types::WEIGHTS, 0 });
        }

        if (use_bias)
        {
            desc.data.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
        }

        return desc;
    }

    std::shared_ptr<KernelString> CommonKernelBase::GetKernelString(std::string name, std::string jit, std::string entry_point, std::string exe_mode) const
    {
        std::shared_ptr<KernelString> kernel_string = std::make_shared<KernelString>();

        auto codes = db.get(name);

        if (codes.size())
        {
            kernel_string->str = codes[0];
            kernel_string->jit = jit;
            kernel_string->options = exe_mode + " -cl-mad-enable";
            kernel_string->entry_point = entry_point;
            kernel_string->batch_compilation = true;
        }

        return kernel_string;
    }

    void CommonKernelBase::FillCLKernelData(clKernelData& kernel, const CommonDispatchData& runInfo, std::string kernel_map_name, std::string jit, std::string entry_point, bool weights, bool bias) const
    {
        kernel.workGroups.global = { runInfo.gws0, runInfo.gws1, runInfo.gws2 };
        kernel.workGroups.local = { runInfo.lws0, runInfo.lws1, runInfo.lws2 };
        kernel.kernelString = GetKernelString(kernel_map_name, jit, entry_point);
        kernel.argsDesc = GetArgsDesc(1, weights, bias);
    }
}