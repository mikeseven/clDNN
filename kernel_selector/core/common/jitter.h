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
#include "tensor_type.h"
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

inline std::string weight_type_2_cl_type(WeightsType wType)
{
    switch (wType)
    {
    case WeightsType::F16: return "half";
    case WeightsType::F32: return "float";
    case WeightsType::INT8: return "char";
    default: return "";
    }
}

inline std::string data_type_2_cl_type(Datatype wType)
{
    switch (wType)
    {
    case Datatype::F16: return "half";
    case Datatype::F32: return "float";
    default: return "";
    }
}

class data_tensor_jit_constant : public jit_constant 
{
    const KernelSelector::DataTensor _tensor;

public:
    data_tensor_jit_constant(const std::string& name, const KernelSelector::DataTensor& t) : jit_constant(name), _tensor(t) {}

    jit_definitions get_definitions() const override 
    {
        jit_definitions definitions{
            { _name + "_TYPE",          data_type_2_cl_type(_tensor.dtype) },
            { _name + "_OFFSET",        std::to_string(_tensor.offset) },
            { _name + "_LIMIT",         std::to_string(_tensor.LengthWithPadding()) },
            { _name + "_DIMS",          std::to_string(_tensor.dims.size()) },
            { _name + "_SIZE_X",        std::to_string(_tensor.x().v) },
            { _name + "_SIZE_Y",        std::to_string(_tensor.y().v) },
            { _name + "_FEATURE_NUM",   std::to_string(_tensor.feature().v) },
            { _name + "_BATCH_NUM",     std::to_string(_tensor.batch().v) },
            { _name + "_X_PITCH",       std::to_string(_tensor.x().pitch) },
            { _name + "_Y_PITCH",       std::to_string(_tensor.y().pitch) },
            { _name + "_FEATURE_PITCH", std::to_string(_tensor.feature().pitch) },
            { _name + "_BATCH_PITCH",   std::to_string(_tensor.batch().pitch) },
            { _name + "_SIMPLE",        std::to_string(_tensor.SimpleLayout()) },
            { "TO_" + _name + "_TYPE",  "convert_" + data_type_2_cl_type(_tensor.dtype) },
            { _name + "_LAYOUT_" + toString(_tensor.layout), "1" },
        };

        definitions.push_back({ _name + "_SIZE", std::to_string(_tensor.dims.size()) });

        // TODO: refactor it
        {
            std::stringstream ss;
            ss << "(size_t []){ ";
            for (size_t i = 0; i < _tensor.dims.size(); i++)
                ss << to_code_string(_tensor.dims[i].v) << ",";
            for (size_t i = _tensor.dims.size(); i < CLDNN_TENSOR_DIM_MAX; i++)
                ss << 1 << ",";
            ss << " } ";
            definitions.push_back({ _name + "_SIZES", ss.str() });
        }
        {
            std::stringstream ss;
            ss << "(size_t []){ ";
            for (size_t i = 0; i < _tensor.dims.size(); i++)
                ss << to_code_string(_tensor.dims[i].pitch) << ",";
            for (size_t i = _tensor.dims.size(); i < CLDNN_TENSOR_DIM_MAX; i++)
                ss << 1 << ",";
            ss << " } ";
            definitions.push_back({ _name + "_PITCHES", ss.str() });
        }

        return definitions;
    }
};

inline std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const KernelSelector::DataTensor& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<data_tensor_jit_constant>(name, value));
}

class weight_tensor_jit_constant : public jit_constant 
{
    const KernelSelector::WeightsTensor _tensor;

public:
    weight_tensor_jit_constant(const std::string& name, const KernelSelector::WeightsTensor& t) : jit_constant(name), _tensor(t) {}

    jit_definitions get_definitions() const override 
    {
        jit_definitions definitions{
            { _name + "_TYPE",          weight_type_2_cl_type(_tensor.wtype) },
            { _name + "_OFFSET",        std::to_string(_tensor.offset) },
            { _name + "_LIMIT",         std::to_string(_tensor.LengthWithPadding()) },
            { _name + "_DIMS",          std::to_string(_tensor.dims.size()) },
            { _name + "_SIZE_X",        std::to_string(_tensor.x().v) },
            { _name + "_SIZE_Y",        std::to_string(_tensor.y().v) },
            { _name + "_IFM_NUM",       std::to_string(_tensor.ifm().v) },
            { _name + "_OFM_NUM",       std::to_string(_tensor.ofm().v) },
            { _name + "_X_PITCH",       std::to_string(_tensor.x().pitch) },
            { _name + "_Y_PITCH",       std::to_string(_tensor.y().pitch) },
            { _name + "_IFM_PITCH",     std::to_string(_tensor.ifm().pitch) },
            { _name + "_OFM_PITCH",     std::to_string(_tensor.ofm().pitch) },
            { _name + "_SIMPLE",        std::to_string(_tensor.SimpleLayout()) },
            { "TO_" + _name + "_TYPE",  "convert_" + weight_type_2_cl_type(_tensor.wtype) },
            { _name + "_LAYOUT_" + toString(_tensor.layout), "1" },
        };

        // TODO: refactor it
        
        definitions.push_back({ _name + "_SIZE", std::to_string(_tensor.dims.size()) });
        
        {
            std::stringstream ss;
            ss << "(size_t []){ ";
            for (size_t i = 0; i < _tensor.dims.size(); i++)
                ss << to_code_string(_tensor.dims[i].v) << ",";
            for (size_t i = _tensor.dims.size(); i < CLDNN_TENSOR_DIM_MAX; i++)
                ss << 1 << ",";
            ss << " } ";
            definitions.push_back({ _name + "_SIZES", ss.str() });
        }
        {
            std::stringstream ss;
            ss << "(size_t []){ ";
            for (size_t i = 0; i < _tensor.dims.size(); i++)
                ss << to_code_string(_tensor.dims[i].pitch) << ",";
            for (size_t i = _tensor.dims.size(); i < CLDNN_TENSOR_DIM_MAX; i++)
                ss << 1 << ",";
            ss << " } ";
            definitions.push_back({ _name + "_PITCHES", ss.str() });
        }

        return definitions;
    }
};

inline std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const KernelSelector::WeightsTensor& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<weight_tensor_jit_constant>(name, value));
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

template <typename T>
inline std::string get_type_name() { throw std::runtime_error("Implement me"); }
template <>
inline std::string get_type_name<double>() { return "double"; }
template <>
inline std::string get_type_name<float>() { return "float"; }
template <>
inline std::string get_type_name<int>() { return "int"; }
template <>
inline std::string get_type_name<unsigned>() { return "unsigned"; }
template <>
inline std::string get_type_name<char>() { return "char"; }
template <>
inline std::string get_type_name<short>() { return "short"; }
template <>
inline std::string get_type_name<uint16_t>() { return "unsigned short"; }

template <typename T>
class vector_data_jit_constant : public jit_constant 
{
    const std::vector<T> _data;

public:
    vector_data_jit_constant(const std::string& name, const std::vector<T>& data) : jit_constant(name), _data(data) {}

    jit_definitions get_definitions() const override 
    {
        std::stringstream ss;
        jit_definitions result;
        result.push_back({ _name + "_SIZE", std::to_string(_data.size()) });
        ss << "(" << get_type_name<T>() << "[]){ ";
        for (size_t i = 0; i < _data.size(); i++)
            ss << to_code_string(_data[i]) << ",";
        ss << " } ";
        
        result.push_back({ _name, ss.str() });
        return result;
    }
};

template <typename T>
inline  std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const std::vector<T> value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<vector_data_jit_constant<T>>(name, value));
}

class jit_constants {
    std::vector<std::shared_ptr<jit_constant>> _constants;
public:
    jit_constants(std::initializer_list<std::shared_ptr<jit_constant>> constants) :_constants(constants) {}

    void add_constant(std::shared_ptr<jit_constant> constant)
    {
        _constants.push_back(constant);
    }

    void add_constants(const std::vector<std::shared_ptr<jit_constant>>& constants)
    {
        for (const auto& c : constants)
        {
            _constants.push_back(c);
        }
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
