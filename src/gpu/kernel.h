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
#include "memory_gpu.h"
#include "kernels_cache.h"
#include "api/instrumentation.h"
#include <iostream>
#include <sstream>

namespace neural { namespace gpu {

class memory_arg : public context_holder {
    const neural::memory& _mem;
    std::shared_ptr<gpu_buffer> _gpu_buffer;
    bool is_own() const {
        return _mem.argument.engine == neural::engine::gpu;
    }
    bool _copy_input;
    bool _copy_output;

protected:
    memory_arg(const neural::memory& mem, bool copy_input, bool copy_output);

public:
    const cl::Buffer& get_buffer() const { return _gpu_buffer->get_buffer(); }
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

// TODO improve to_code_string specializations
template<typename T>
std::string to_code_string(T val) { return std::to_string(val); }

template<>
inline std::string to_code_string<std::string>(std::string val) { return val; }

template<>
inline std::string to_code_string<const char*>(const char* val) { return val; }

template<>
inline std::string to_code_string<char*>(char* val) { return val; }

template<>
inline std::string to_code_string<float>(float val) {
    // 64 chars should be enought to store: "-0x0.123456p-123f /*-0.123456e-123*/"
    char buffer[64] = "";
    std::snprintf(buffer, sizeof(buffer), "%.6af /*%.4g*/", double(val), double(val));
    return buffer;
}

template<>
inline std::string to_code_string<double>(double val) {
    // 64 chars should be enought to store: "-0x0.1234567890123p-1234 /*-0.1234567890123e-1074*/"
    char buffer[64] = "";
    std::snprintf(buffer, sizeof(buffer), "%.13a /*%.4g*/", val, val);
    return buffer;
}

// TODO refactor jit_constant, make_jit_constant, etc...
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

template<typename T>
std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, T value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<simple_jit_constant>(name, to_code_string(value)));
}

template<typename T>
class vector_jit_constant : public jit_constant {
    const neural::vector<T> _vec;

public:
    vector_jit_constant(const std::string& name, const neural::vector<T>& vec)
        : jit_constant(name), _vec(vec) {}

    kernels_cache::jit_definitions get_definitions() const override {

        kernels_cache::jit_definitions definitions{
            { _name + "_BATCH_NUM", std::to_string(_vec.batch[0]) },
        };

        const char* spatial_names[] = { "X", "Y", "Z", "W" };
        if (_vec.spatial.size() > std::size(spatial_names))
            throw std::runtime_error("max 4D images are supported");

        for (size_t i = 0; i < std::max(_vec.spatial.size(), static_cast<size_t>(2)); ++i) {
            definitions.emplace_back( _name + "_SIZE_" + spatial_names[i],
                                      _vec.spatial.size() > i ? std::to_string(_vec.spatial[i]) : "1" );
        }

        assert(_vec.feature.size() > 0);
        if (_vec.feature.size() > 0) {
            // if number of feature nums is 1 then no suffix
            if(_vec.feature.size() == 1) {
                definitions.emplace_back(_name + "_FEATURE_NUM", std::to_string(_vec.feature[0]));
            }
            else { // else add suffixes
                for (size_t i = 0; i < _vec.feature.size(); ++i) {
                    definitions.emplace_back(_name + "_FEATURE_NUM_" + std::to_string(i), std::to_string(_vec.feature[i]));
                }
            }
        }
        return definitions;
    }
};

template<typename T>
inline std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const neural::vector<T>& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<vector_jit_constant<T>>(name, value));
}

class memory_jit_constant : public vector_jit_constant<uint32_t> {
    const neural::memory& _mem;

public:
    memory_jit_constant(const std::string& name, const neural::memory& mem)
        : vector_jit_constant(name, mem.argument.size), _mem(mem){}

    kernels_cache::jit_definitions get_definitions() const override {
        auto result = vector_jit_constant::get_definitions();
        auto data = _mem.pointer<float>();
        std::stringstream ss;
        ss << "(float[]){ ";
        for (size_t i = 0; i < _mem.count(); i++)
            ss << to_code_string(data[i]) << ",";
        ss << " } ";
        result.push_back({ _name, ss.str() });
        return result;
    }
};

inline  std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const neural::memory& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<memory_jit_constant>(name, value));
}

class memories_jit_constant : public vector_jit_constant<uint32_t> {
    const std::vector<std::reference_wrapper<const neural::memory>> _mem;

public:
    memories_jit_constant(const std::string& name, const std::vector<std::reference_wrapper<const neural::memory>> mem)
        :vector_jit_constant(name, mem[0].get().argument.size), _mem(mem) {}

    kernels_cache::jit_definitions get_definitions() const override {
        for (size_t i = 1; i < _mem.size(); i++)
        {
            if (_mem[0].get().count() != _mem[i].get().count())
                throw std::exception("All memories must contain the same number of elements!");
        }
        auto result = vector_jit_constant::get_definitions();
        result.push_back({ _name + "_ARRAY_NUM", std::to_string(_mem.size()) });
        std::stringstream ss;
        ss << "(float[][" + std::to_string(_mem[0].get().count()) + "]) {";
        for (auto& m : _mem)
        {
            auto & _m = m.get();
            auto data = _m.pointer<float>();
            ss << "{ ";
            for (size_t i = 0; i < _m.count(); i++)
                ss << to_code_string(data[i]) << ",";
            ss << " } ,";
        }
        ss << " } ";
        result.push_back({ _name, ss.str() });
        return result;
    }
};

inline  std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const std::vector<std::reference_wrapper<const neural::memory>> value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<memories_jit_constant>(name, value));
}

class jit_constants {
    std::vector<std::shared_ptr<jit_constant>> _constants;
public:
    jit_constants(std::initializer_list<std::shared_ptr<jit_constant>> constants) :_constants(constants) {}

    void add_constant(std::shared_ptr<jit_constant> constant)
    {
        _constants.push_back(constant);
    }

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

class kernel : public context_holder {
    kernels_cache::kernel_id _kernel_id;

    template<unsigned index, typename Ti, typename... Ts>
    void setArgs(cl::Kernel& clkernel, Ti&& arg, Ts&&... args) const {
        clkernel.setArg(index, kernel_arg_handler<Ti>::get(arg));
        setArgs<index + 1, Ts...>(clkernel, std::forward<Ts>(args)...);
    }


    template<unsigned index, typename Ti>
    void setArgs(cl::Kernel& clkernel, Ti&& arg) const {
        clkernel.setArg(index, kernel_arg_handler<Ti>::get(arg));
    }

    template<unsigned index>
    void setArgs(cl::Kernel&) const {}

public:
    explicit kernel(const std::string& name, kernels_cache::jit_definitions definitions = kernels_cache::jit_definitions())
        : _kernel_id(kernels_cache::get().create_kernel_from_template(context(), name, definitions)) {}
    explicit kernel(const std::string& name, const jit_constants& constants) 
        : _kernel_id(kernels_cache::get().create_kernel_from_template(context(), name, constants.get_definitions())) {}

    kernel(const kernel& other) : _kernel_id(other._kernel_id) {}

    kernel& operator=(const kernel& other) {
        if (this == &other)
            return *this;
        _kernel_id = other._kernel_id;
        return *this;
    }

    template<typename... Args>
    void run(const kernel_execution_options& options, Args... args) const {
        if (configuration::get().enable_profiling) {
            instrumentation::timer<> pre_enqueue_timer;
            auto clkernel = kernels_cache::get().get_kernel(context(), _kernel_id);
            setArgs<0>(clkernel, std::forward<Args>(args)...);
            auto pre_enqueue_time = pre_enqueue_timer.uptime();
            cl::Event end_event;
            context()->queue().enqueueNDRangeKernel(clkernel, cl::NullRange, options.global_range(), options.local_range(), 0, &end_event);
            end_event.wait();
            context()->report_profiling({ _kernel_id,
                {
                    {"pre-enqueue", std::make_shared<instrumentation::profiling_period_basic>(pre_enqueue_time)},
                    {"submission",  std::make_shared<profiling_period_event>(end_event, CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_SUBMIT)},
                    {"starting",    std::make_shared<profiling_period_event>(end_event, CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_START)},
                    {"executing",   std::make_shared<profiling_period_event>(end_event, CL_PROFILING_COMMAND_START,  CL_PROFILING_COMMAND_END)}
                } });
        }
        else {
            auto clkernel = kernels_cache::get().get_kernel(context(), _kernel_id);
            setArgs<0>(clkernel, std::forward<Args>(args)...);
            context()->queue().enqueueNDRangeKernel(clkernel, cl::NullRange, options.global_range(), options.local_range());
        }
    }
};

} }
