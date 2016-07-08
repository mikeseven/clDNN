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
#include "ocl_toolkit.h"
#include "kernels_cache.h"
#include <iostream>

namespace neural { namespace gpu {

class vector_arg : public context_holder {
    const neural::vector<uint32_t>& _vec;
    cl::Buffer _clBuffer;
public:
    vector_arg(const neural::vector<uint32_t>& arg);
    const cl::Buffer& get_buffer() const { return _clBuffer; };

    ~vector_arg();
};

class memory_arg : public context_holder {
    const neural::memory& _mem;
    cl::Buffer _clBuffer;
    bool is_own() const {
        return _mem.argument.engine == neural::engine::gpu && _mem.argument.owns_memory;
    }
    bool _copy_input;
    bool _copy_output;

protected:
    memory_arg(const neural::memory& mem, bool copy_input, bool copy_output);

public:
    const cl::Buffer& get_buffer() const { return _clBuffer; };
    ~memory_arg();
};

class input_mem : public memory_arg {
public:
    input_mem(const neural::memory& mem) :memory_arg(mem, true, false) {}
};

class output_mem : public memory_arg {
public:
    output_mem(const neural::memory& mem) :memory_arg(mem, false, true) {}
};

class jit_constant {
protected:
    const std::string _name;
    jit_constant(const std::string& name):_name(name){}

public:
    virtual kernels_cache::jit_definitions get_definitions() const = 0;
    virtual ~jit_constant() {}
};

class simple_jit_constant : public jit_constant {
    const std::string _value;

public:
    simple_jit_constant(const std::string& name, const std::string& value)
        :jit_constant(name), _value(value) {}

    kernels_cache::jit_definitions get_definitions() const override {
        return kernels_cache::jit_definitions{ {_name, _value} };
    }
};

inline std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const std::string& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<simple_jit_constant>(name, value));
}

class vector_jit_constant : public jit_constant {
    const neural::vector<uint32_t>& _vec;

public:
    vector_jit_constant(const std::string& name, const neural::vector<uint32_t>& vec)
        : jit_constant(name), _vec(vec) {}

    kernels_cache::jit_definitions get_definitions() const override {
        auto feature_offset = _vec.batch.size();
        auto spatial_offset = _vec.feature.size() + feature_offset;
        return kernels_cache::jit_definitions{
            { _name + "_BATCH_NUM", std::to_string(_vec.raw[0]) },
            { _name + "_SIZE_X", std::to_string(_vec.raw[0 + spatial_offset]) },
            { _name + "_SIZE_Y", _vec.spatial.size() > 1 ? std::to_string(_vec.raw[1 + spatial_offset]) : "1" },
            { _name + "_OUTPUT_FEATURE_NUM", std::to_string(_vec.raw[0 + feature_offset]) },
            { _name + "_INPUT_FEATURE_NUM", _vec.feature.size() > 1 ? std::to_string(_vec.raw[1 + feature_offset]) : "1" }
        };
    }
};

inline std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const neural::vector<uint32_t>& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<vector_jit_constant>(name, value));
}

class memory_jit_constant : public vector_jit_constant {
    const neural::memory& _mem;

public:
    memory_jit_constant(const std::string& name, const neural::memory& mem)
        : vector_jit_constant(name, mem.argument.size), _mem(mem){}

    kernels_cache::jit_definitions get_definitions() const override {
        auto result = vector_jit_constant::get_definitions();
        std::stringstream ss;
        ss << "(float[]){ ";
        for (int i = 0; i < _mem.count(); i++)
            ss << static_cast<float*>(_mem.pointer)[i] << ",";
        ss << " } ";
        result.push_back({ _name, ss.str() });
        return result;
    }
};

inline  std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const neural::memory& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<memory_jit_constant>(name, value));
}

class jit_constants {
    std::vector<std::shared_ptr<jit_constant>> _constants;
public:
    jit_constants(std::initializer_list<std::shared_ptr<jit_constant>> constants) :_constants(constants) {}

    kernels_cache::jit_definitions get_definitions() const {
        kernels_cache::jit_definitions definitons;
        definitons.reserve(_constants.size() * 6); //assuming max 6 pairs per jit_constant

        for (auto& constant : _constants) {
            auto def = constant->get_definitions();
            definitons.insert(definitons.end(), def.begin(), def.end());
        }
        return definitons;
    }
};

template<typename T, class Enable = void>
struct kernel_arg_handler;

template<typename T>
struct kernel_arg_handler<T, typename std::enable_if<!std::is_base_of<memory_arg, T>::value>::type> {
    static const T& get(const T& arg) { return arg; }
};

template<typename T>
struct kernel_arg_handler<T, typename std::enable_if<std::is_base_of<memory_arg, T>::value>::type> {
    static const cl::Buffer& get(const T& arg) { return arg.get_buffer(); }
};

template<>
struct kernel_arg_handler<vector_arg> {
    static const cl::Buffer& get(const vector_arg& arg) { return arg.get_buffer(); };
};


class kernel_execution_options {
    cl::NDRange _global;
    cl::NDRange _local;
public:
    class range2d {
        size_t x; size_t y;
    public:
        range2d(size_t x, size_t y) :x(x), y(y) {}
        operator cl::NDRange() const { return cl::NDRange(x, y); }
    };

    class range3d {
        size_t x; size_t y; size_t z;
    public:
        range3d(size_t x, size_t y, size_t z) :x(x), y(y), z(z) {}
        operator cl::NDRange() const { return cl::NDRange(x, y, z); }
    };

    kernel_execution_options(size_t work_items, size_t parallel_items) : _global(work_items), _local(parallel_items) {}
    kernel_execution_options(range2d work_items, range2d parallel_items) : _global(work_items), _local(parallel_items){}
    kernel_execution_options(range3d work_items, range3d parallel_items) : _global(work_items), _local(parallel_items) {}

    cl::NDRange global_range() const { return _global; }
    cl::NDRange local_range() const { return _local; }
};

template<typename... Args>
class kernel : public context_holder {
    kernels_cache::kernel_id _kernel_id;
    cl::Kernel _kernel;

    template<unsigned index, typename Ti, typename... Ts>
    void setArgs(Ti&& arg, Ts&&... args) {
        _kernel.setArg(index, kernel_arg_handler<Ti>::get(arg));
        setArgs<index + 1, Ts...>(std::forward<Ts>(args)...);
    }


    template<unsigned index, typename Ti>
    void setArgs(Ti&& arg) {
        _kernel.setArg(index, kernel_arg_handler<Ti>::get(arg));
    }

    template<unsigned index>
    void setArgs() {}

public:
    explicit kernel(const std::string& name, kernels_cache::jit_definitions definitions = kernels_cache::jit_definitions())
        : _kernel_id(kernels_cache::get().create_kernel_from_template(context(), name, definitions)) {}
    explicit kernel(const std::string& name, const jit_constants& constants) 
        : _kernel_id(kernels_cache::get().create_kernel_from_template(context(), name, constants.get_definitions())) {}

    void operator()(const kernel_execution_options& options, Args... args) {
        _kernel = kernels_cache::get().get_kernel(context(), _kernel_id);
        setArgs<0>(std::forward<Args>(args)...);

        try {
            cl::Event end_event;
            context()->queue().enqueueNDRangeKernel(_kernel, cl::NullRange, options.global_range(), options.local_range(), 0, &end_event);
            end_event.wait();
        } catch(cl::Error err) {
            std::cerr << "ERROR:" << err.what() << std::endl;
        }
    }
};

} }
