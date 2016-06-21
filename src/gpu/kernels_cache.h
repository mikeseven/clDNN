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
    static std::mutex _mutex;
    static std::map<std::string, std::string> _templates;
public:
    static void add(const std::string& name, const std::string& code) {
        std::lock_guard<std::mutex> lock(_mutex);
        _templates.insert({ name, code });
    }
    static std::string get(const std::string& name) {
        std::lock_guard<std::mutex> lock(_mutex);
        std::string result = _templates.at(name);
        return result;
    }
};

class kernels_cache {
public:
    typedef std::string kernel_id;
private:
    std::mutex _mutex;
    std::map<std::string, std::string> _kernel_codes;
    bool _modified = true;

    cl::Program::Sources get_program_source() const;
    kernels_cache() = default;

public:
    static kernels_cache& get();
    bool modified() const { return _modified; }

    kernel_id create_kernel_from_template(const std::string& template_name, std::map<std::string, std::string> definitions = std::map<std::string, std::string>());
    cl::Program get_program(neural::gpu::gpu_toolkit* context);
    cl::Kernel get_kernel(neural::gpu::gpu_toolkit* context, kernel_id id);
};

}}
