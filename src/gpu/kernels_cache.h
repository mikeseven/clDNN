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
#include <map>
#include <mutex>
#include <vector>
#include "cache/primitive_db.h"

namespace cl {
class Kernel;
}

namespace neural {namespace gpu {
class gpu_toolkit;

class kernels_cache {
public:
    using source_code = std::vector<std::string>;

    struct program_code
    {
        source_code source;
        std::string options;
        bool dump_custom_program;
    };

    struct kernel_code
    {
        source_code source;
        std::string options;
        bool batch_compilation;
        bool inject_header;
        bool dump_custom_program;
    };

    typedef std::string kernel_id;
    typedef std::vector<std::pair<std::string, std::string>> jit_definitions;
    typedef cl::Kernel kernel_type;
    using sorted_code = std::map<std::string, program_code>;
    using kernels_map = std::map<std::string, kernel_type>;
    using kernels_code = std::map<std::string, kernel_code>;

private:
    gpu_toolkit& _context;
    std::mutex _mutex;
    kernels_code _kernels_code;
    std::map<std::string, kernel_type> _kernels;
    manager::primitive_db _database;
    bool _modified = true;

    sorted_code get_program_source(const kernels_code& kernels_source_code) const;
    friend class gpu_toolkit;
    explicit kernels_cache(gpu_toolkit& context);
    kernels_map build_program(const program_code& pcode) const;

public:
    kernel_id create_kernel_from_template(const std::string& template_name, jit_definitions definitions = jit_definitions(), std::string kernel_name = std::string());
    kernel_id set_kernel_source(const std::string& template_name, jit_definitions definitions = jit_definitions(), std::string kernel_name = std::string());
    kernel_id set_kernel_source(const source_code& source, const std::string& options, const std::string& entry_point, bool batch_compilation, bool dump_custom_program);
    kernel_type get_kernel(kernel_id id);
};

}}
