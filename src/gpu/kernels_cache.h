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

private:
    std::mutex _mutex;
    std::map<std::string, std::string> _kernel_codes;
    bool _modified = true;

    cl::Program::Sources get_program_source() const;
    kernels_cache() = default;
    cl::Program get_program(std::shared_ptr<neural::gpu::gpu_toolkit> context);

public:
    static kernels_cache& get();

    kernel_id create_kernel_from_template(std::shared_ptr<neural::gpu::gpu_toolkit> context, const std::string& template_name, jit_definitions definitions = jit_definitions());
    cl::Kernel get_kernel(std::shared_ptr<neural::gpu::gpu_toolkit> context, kernel_id id);
};

}}
