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

#include "cldnn_defs.h"
#include "compounds.h"
#include "tensor.hpp"

#include <algorithm>
#include <string>
#include <vector>


namespace cldnn
{
struct padding
{
    float filling_value() const { return _filling_value; }
    
    /// Gets lower padding sizes. For spatials, it means size of left (X) and top (Y) padding.
    ///
    /// @return Tensor with padding for top/left/lower bounds of data.
    tensor lower_size() const { return _lower_size; }
    /// Gets upper padding sizes. For spatials, it means size of right (X) and bottom (Y) padding.
    ///
    /// @return Tensor with padding for bottom/right/upper bounds of data.
    tensor upper_size() const { return _upper_size; }
    /// Gets format of tensors used in padding.
    cldnn::format format() const { return _lower_size.format; }

    padding(cldnn::format format, const std::vector<tensor::value_type>& lower_sizes, const std::vector<tensor::value_type>& upper_sizes, float filling_value = 0.0f)
        : _lower_size(format, 0, to_abs(lower_sizes)), _upper_size(format, 0, to_abs(upper_sizes)), _filling_value(filling_value)
    {}

    padding(cldnn::format format, const std::vector<tensor::value_type>& sizes, float filling_value = 0.0f)
        : padding(format, sizes, sizes, filling_value)
    {}

    padding(): padding(format::x, {0}) {}

    padding(const cldnn_padding& other)
        : _lower_size(other.lower_size), _upper_size(other.upper_size), _filling_value(other.filling_value)
    {}

    operator cldnn_padding() const
    {
        return { static_cast<cldnn_tensor>(_lower_size),
                 static_cast<cldnn_tensor>(_upper_size),
                 _filling_value };
    }

    // returns true if padding size is not zero
    explicit operator bool() const
    {
        return std::any_of(_lower_size.raw.begin(), _lower_size.raw.end(), [](const tensor::value_type& el) { return el != 0; }) ||
               std::any_of(_upper_size.raw.begin(), _upper_size.raw.end(), [](const tensor::value_type& el) { return el != 0; });
    }

    static padding max(padding const& lhs, padding const& rhs, float filling_value = 0.0f)
    {
        return padding{ tensor::max(lhs.lower_size(), rhs.lower_size()),
                        tensor::max(lhs.upper_size(), rhs.upper_size()),
                        filling_value };
    }


private:
    tensor _lower_size;    ///< Lower padding sizes. For spatials, it means size of left (X) and top (Y) padding.
    tensor _upper_size;    ///< Upper padding sizes. For spatials, it means size of right (X) and bottom (Y) padding.
    // TODO: Add support for non-zero filling value (if necessary) or remove variable (if not necessary).
    float  _filling_value; ///< Filling value for an element of padding. If data type of elements is different than float it is converted
                           ///< to it using round-towards-nearest-even (for floating-point data types) or round-towards-zero (for integral
                           ///< data types).

    padding(tensor const& lower, tensor const& upper, float filling_value = 0.0f)
        : _lower_size(lower), _upper_size(upper), _filling_value(filling_value)
    {}

    static std::vector<tensor::value_type> to_abs(const std::vector<tensor::value_type>& sizes)
    {
        std::vector<tensor::value_type> result;
        result.reserve(sizes.size());
        std::transform(sizes.cbegin(), sizes.cend(), std::back_inserter(result), [](const tensor::value_type& el) { return abs(el); });
        return result; // NRVO
    }
};

CLDNN_API_CLASS(padding)

// TODO check this:
using primitive_type_id = cldnn_primitive_type_id;
using primitive_id_ref = cldnn_primitive_id;
using primitive_id = std::string;

template<class PType>
typename PType::dto* as_dto(CLDNN_PRIMITIVE_DESC(primitive)* dto)
{
    if (dto->type != PType::type_id()) throw std::invalid_argument("type");
    return reinterpret_cast<typename PType::dto*>(dto);
}
template<class PType>
const typename PType::dto* as_dto(const CLDNN_PRIMITIVE_DESC(primitive)* dto)
{
    if (dto->type != PType::type_id()) throw std::invalid_argument("type");
    return reinterpret_cast<const typename PType::dto*>(dto);
}

struct primitive
{
    primitive(
        const primitive_type_id& type,
        const primitive_id& id,
        const std::vector<primitive_id>& input,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :_type(type), _id(id), _input(input), _input_padding(input_padding), _output_padding(output_padding)
    {}

    primitive(const CLDNN_PRIMITIVE_DESC(primitive)* dto)
        :_type(dto->type), _id(dto->id), _input(dto->input), _input_padding(dto->input_padding), _output_padding(dto->output_padding)
    {}

    virtual ~primitive() = default;

    const primitive_type_id& type() const { return _type; }
    const primitive_id& id() const { return _id; }

    //TODO: make access to primitive's fields consistent - either use access directly via public members,
    // like in derived classes, or use getters/setter in derived classes like below
    std::vector<primitive_id>& input() { return _input.cpp_ids; }
    const std::vector<primitive_id>& input() const { return _input.cpp_ids; }

    padding& input_padding() { return _input_padding; }
    const padding& input_padding() const { return _input_padding; }

    padding& output_padding() { return _output_padding; }
    const padding& output_padding() const { return _output_padding; }

    virtual const CLDNN_PRIMITIVE_DESC(primitive)* get_dto() const = 0;

    std::vector<primitive_id> dependecies() const
    {
        auto result = input();
        auto deps = get_dependencies();
        result.insert(result.end(), deps.begin(), deps.end());
        return result;
    }

    operator primitive_id() const { return id(); }

    //TODO remove backward compatibility
    tensor input_offset() const { return input_padding().lower_size().negate(); }
    tensor output_offset() const { return output_padding().lower_size(); }
    float padding_filling_value() const { return input_padding().filling_value(); }

protected:
    struct primitive_id_arr
    {
        primitive_id_arr(std::vector<primitive_id> const& vec) : cpp_ids(vec)
        {}

        primitive_id_arr(std::vector<primitive_id>&& vec) : cpp_ids(std::move(vec))
        {}

        //create from C API id array
        primitive_id_arr(cldnn_primitive_id_arr c_id_arr)
        {
            cpp_ids.resize(c_id_arr.size);
            for (size_t i = 0; i < c_id_arr.size; ++i)
                cpp_ids[i] = c_id_arr.data[i];
        }

        std::vector<primitive_id> cpp_ids;
        mutable std::vector<cldnn_primitive_id> c_ids;
        //get C API id array
        auto ref() const -> decltype(cldnn_primitive_id_arr{c_ids.data(), c_ids.size()})
        {
            c_ids.resize(cpp_ids.size());
            for (size_t i = 0; i < cpp_ids.size(); ++i)
                c_ids[i] = cpp_ids[i].c_str();

            return cldnn_primitive_id_arr{ c_ids.data(), c_ids.size() };
        }

        size_t size() const { return cpp_ids.size(); }



    };

    const primitive_type_id _type;
    const primitive_id _id;

    primitive_id_arr _input;
    padding _input_padding;
    padding _output_padding;

    virtual std::vector<primitive_id> get_dependencies() const { return{}; }
};

template<class PType, class DTO>
class primitive_base : public primitive
{
public:
    const CLDNN_PRIMITIVE_DESC(primitive)* get_dto() const override
    {
        //update common dto fields
        _dto.id = _id.c_str();
        _dto.type = _type;
        _dto.input = _input.ref();
        _dto.input_padding = _input_padding;
        _dto.output_padding = _output_padding;

        //call abstract method to update primitive-specific fields
        update_dto(_dto);
        return reinterpret_cast<const CLDNN_PRIMITIVE_DESC(primitive)*>(&_dto);
    }

protected:
    explicit primitive_base(
        const primitive_id& id,
        const std::vector<primitive_id>& input,
        const padding& input_padding = padding(),
        const padding& output_padding = padding())
        : primitive(PType::type_id(), id, input, input_padding, output_padding)
    {}

    primitive_base(const DTO* dto)
        : primitive(reinterpret_cast<const CLDNN_PRIMITIVE_DESC(primitive)*>(dto))
    {
        if (dto->type != PType::type_id()) 
            throw std::invalid_argument("DTO type mismatch");
    }

private:
    mutable DTO _dto;

    virtual void update_dto(DTO& dto) const = 0;
};

#define CLDNN_DEFINE_TYPE_ID(PType) static primitive_type_id type_id()\
    {\
        return check_status<primitive_type_id>( #PType " type id failed", [](status_t* status)\
        {\
            return cldnn_##PType##_type_id(status);\
        });\
    }

#define CLDNN_DECLATE_PRIMITIVE(PType) typedef CLDNN_PRIMITIVE_DESC(PType) dto;\
    CLDNN_DEFINE_TYPE_ID(PType)

}
