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

namespace cl {
class Kernel;
}

namespace neural {namespace gpu {
class gpu_toolkit;

class kernel_templates {
    std::mutex _mutex;
    std::map<std::string, std::string> _templates;
    kernel_templates(){}
    static kernel_templates& instance() {
        static kernel_templates instance;
        return instance;
    }
public:
    kernel_templates(const kernel_templates& other) = delete;
    kernel_templates(kernel_templates&& other) = delete;
    kernel_templates& operator=(const kernel_templates& other) = delete;
    kernel_templates& operator=(kernel_templates&& other) = delete;

    static void add(const std::string& name, const std::string& code) {
        std::lock_guard<std::mutex> lock(instance()._mutex);
        instance()._templates.insert({ name, code });
    }
    static std::string get(const std::string& name) {
        std::lock_guard<std::mutex> lock(instance()._mutex);
        std::string result = instance()._templates.at(name);
        return result;
    }

};

class kernels_cache {
public:
    typedef std::string kernel_id;
    typedef std::vector<std::pair<std::string, std::string>> jit_definitions;
    typedef cl::Kernel kernel_type;

private:
    gpu_toolkit& _context;
    std::mutex _mutex;
    std::map<std::string, std::string> _kernel_codes;
    std::map<std::string, kernel_type> _kernels;
    bool _modified = true;

    std::vector<std::string> get_program_source() const;
    friend class gpu_toolkit;
    explicit kernels_cache(gpu_toolkit& context): _context(context){}
    void build_program();

public:
    kernel_id create_kernel_from_template(const std::string& template_name, jit_definitions definitions = jit_definitions());
    kernel_type get_kernel(kernel_id id);
};

}}
