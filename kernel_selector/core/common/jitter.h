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
#include "api/CPP/memory.hpp"
#include "api/CPP/tensor.hpp"
#include "api/CPP/profiling.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

namespace KernelSelector { namespace gpu {

typedef std::vector<std::pair<std::string, std::string>> jit_definitions;

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
    if (std::isinf(val))
        std::snprintf(buffer, sizeof(buffer), "%sINFINITY", std::signbit(val) ? "-" : "");
    else
        std::snprintf(buffer, sizeof(buffer), "%.6af /*%.4g*/", double(val), double(val));
    return buffer;
}

template<>
inline std::string to_code_string<double>(double val) {
    // 64 chars should be enought to store: "-0x0.1234567890123p-1234 /*-0.1234567890123e-1074*/"
    char buffer[64] = "";
    if (std::isinf(val))
        std::snprintf(buffer, sizeof(buffer), "%sINFINITY", std::signbit(val) ? "-" : "");
    else
        std::snprintf(buffer, sizeof(buffer), "%.13a /*%.4g*/", val, val);
    return buffer;
}

// TODO refactor jit_constant, make_jit_constant, etc...
class jit_constant {
protected:
    const std::string _name;
    jit_constant(const std::string& name):_name(name){}

public:
    virtual jit_definitions get_definitions() const = 0;
    virtual ~jit_constant() {}
};

class simple_jit_constant : public jit_constant {
    const std::string _value;

public:
    simple_jit_constant(const std::string& name, const std::string& value)
        :jit_constant(name), _value(value) {}

    jit_definitions get_definitions() const override {
        return jit_definitions{ {_name, _value} };
    }
};

template<typename T>
std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, T value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<simple_jit_constant>(name, to_code_string(value)));
}

class vector_jit_constant : public jit_constant {
    const cldnn::tensor _vec;

public:
    vector_jit_constant(const std::string& name, const cldnn::tensor& vec)
        : jit_constant(name), _vec(vec) {}

    jit_definitions get_definitions() const override {

        jit_definitions definitions{
            { _name + "_BATCH_NUM", std::to_string(_vec.batch[0]) },
        };

        const char* spatial_names[] = { "X", "Y", "Z", "W" };
        if (_vec.spatial.size() > (sizeof(spatial_names)/sizeof(spatial_names[0])))
            throw std::runtime_error("max 4D images are supported");

        // set default spatial value to "1"
        cldnn::tensor::value_type spatial_value = 1;
        for (size_t i = 0; i < std::max(_vec.spatial.size(), static_cast<size_t>(2)); ++i) {
            // tensor's spatials num is less than 2
            //      then use the value of the last spatial (or default "1")
            if (_vec.spatial.size() > i)
                spatial_value = _vec.spatial[i];
            definitions.emplace_back( _name + "_SIZE_" + spatial_names[i], std::to_string(spatial_value));
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

inline std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const cldnn::tensor& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<vector_jit_constant>(name, value));
}

class padding_jit_constant : public jit_constant {
    vector_jit_constant _lower_size_jit;
    vector_jit_constant _upper_size_jit;

public:
    padding_jit_constant(const std::string& name, const cldnn::padding& pad)
        : jit_constant(name),
          _lower_size_jit(name + "_LOWER", pad.lower_size()),
          _upper_size_jit(name + "_UPPER", pad.upper_size()) {}

    jit_definitions get_definitions() const override {
        auto&& lower_jits = _lower_size_jit.get_definitions();
        auto&& upper_jits = _upper_size_jit.get_definitions();
        lower_jits.insert(lower_jits.cend(), upper_jits.cbegin(), upper_jits.cend());

        return lower_jits;
    }
};

inline std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const cldnn::padding& value) {
    return std::make_shared<padding_jit_constant>(name, value);
}

class memory_jit_constant : public vector_jit_constant {
    const cldnn::memory _mem;

public:
    memory_jit_constant(const std::string& name, const cldnn::memory& mem)
        : vector_jit_constant(name, mem.get_layout().size), _mem(mem){}

    jit_definitions get_definitions() const override {
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

inline  std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const cldnn::memory& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<memory_jit_constant>(name, value));
}



class memories_jit_constant : public vector_jit_constant {
    const std::vector<cldnn::memory> _mem;

public:
    memories_jit_constant(const std::string& name, const std::vector<cldnn::memory> mem)
        :vector_jit_constant(name, mem[0].get_layout().size), _mem(mem) {}

    jit_definitions get_definitions() const override {
        for (size_t i = 1; i < _mem.size(); i++)
        {
            if (_mem[0].count() != _mem[i].count())
                throw std::invalid_argument("All memories must contain the same number of elements!");
        }
        auto result = vector_jit_constant::get_definitions();
        result.push_back({ _name + "_ARRAY_NUM", std::to_string(_mem.size()) });
        std::stringstream ss;
        ss << "(float[][" + std::to_string(_mem[0].count()) + "]) {";
        for (auto& m : _mem)
        {
            auto data = m.pointer<float>();
            ss << "{ ";
            for (size_t i = 0; i < m.count(); i++)
                ss << to_code_string(data[i]) << ",";
            ss << " } ,";
        }
        ss << " } ";
        result.push_back({ _name, ss.str() });
        return result;
    }
};

inline  std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const std::vector<cldnn::memory> value) {
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

    jit_definitions get_definitions() const {
        jit_definitions definitons;
        definitons.reserve(_constants.size() * 6); //assuming max 6 pairs per jit_constant

        for (auto& constant : _constants) {
            auto def = constant->get_definitions();
            definitons.insert(definitons.end(), def.begin(), def.end());
        }
        return definitons;
    }
};

} }
