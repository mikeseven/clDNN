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

private:
    tensor _lower_size;    ///< Lower padding sizes. For spatials, it means size of left (X) and top (Y) padding.
    tensor _upper_size;    ///< Upper padding sizes. For spatials, it means size of right (X) and bottom (Y) padding.
    // TODO: Add support for non-zero filling value (if necessary) or remove variable (if not necessary).
    float  _filling_value; ///< Filling value for an element of padding. If data type of elements is different than float it is converted
                           ///< to it using round-towards-nearest-even (for floating-point data types) or round-towards-zero (for integral
                           ///< data types).

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

namespace details
{
class primitive_id_arr
{
public:
    primitive_id_arr(const std::vector<primitive_id>& arr)
        : _data_store(arr)
    {
        update_ref();
    }

    primitive_id_arr(std::vector<primitive_id>&& arr)
        : _data_store(std::move(arr))
    {
        update_ref();
    }

    primitive_id_arr(const cldnn_primitive_id_arr& arr)
        : _data_store(create_store(arr))
    {
        update_ref();
    }

    primitive_id_arr(primitive_id_arr&& other)
        : _data_store(std::move(other)), _ref_store(std::move(other._ref_store))
    {}

    primitive_id_arr(const primitive_id_arr& other)
        : _data_store(other)
    {
        update_ref();
    }

    //allow only const functions
    auto size() const { return _data_store.size(); }
    decltype(auto) back() const { return _data_store.back(); }
    decltype(auto) front() const { return _data_store.front(); }
    decltype(auto) at(size_t idx) const { return _data_store.at(idx); }
    decltype(auto) operator[](size_t idx) const { return _data_store.operator[](idx); }
    auto data() const { return _data_store.data(); }
    auto begin() const { return _data_store.begin(); }
    auto end() const { return _data_store.end(); }
    auto cbegin() const { return _data_store.cbegin(); }
    auto cend() const { return _data_store.cend(); }

    //allow modifications through following functions (keeps synchronisation with C API storage)
    void update(size_t pos, primitive_id const& pid)
    {
        assert(pos < size() && "Array element out of range");
        if (pos >= size())
            return;
        _data_store[pos] = pid;
        _ref_store[pos] = at(pos).c_str();
    }

    void update(size_t pos, primitive_id&& pid)
    {
        assert(pos < size() && "Array element out of range");
        if (pos >= size())
            return;
        _data_store[pos] = std::move(pid);
        _ref_store[pos] = at(pos).c_str();
    }

    void erase(size_t pos)
    {
        assert(pos < size() && "Array element out of range");
        if (pos >= size())
            return;
        _data_store.erase(begin() + pos);
        _ref_store.erase(_ref_store.begin() + pos);
    }

    void push_back(primitive_id const& pid)
    {
        _data_store.push_back(pid);
        _ref_store.push_back(back().c_str());
    }

    // explicit conversion
    cldnn_primitive_id_arr ref() const
    {
        return { _ref_store.data(), _ref_store.size() };
    }

    auto const& store() const
    {
        return _data_store;
    }

    operator std::vector<primitive_id> const&() const
    {
        return _data_store;
    }

private:
    std::vector<primitive_id> _data_store;
    std::vector<cldnn_primitive_id> _ref_store;

    void update_ref()
    {
        _ref_store.resize(size());
        for (size_t i = 0; i < size(); ++i)
            _ref_store[i] = at(i).c_str();
    }

    static std::vector<primitive_id> create_store(const cldnn_primitive_id_arr& arr)
    {
        //Fill _data_store by copies of strings referenced in arr
        std::vector<primitive_id> result;
        result.reserve(arr.size);
        for (size_t i = 0; i < arr.size; i++)
        {
            result.push_back(arr.data[i]);
        }
        return std::move(result);
    }
};
} // namespace detail

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

    auto& input() { return _input; }
    const auto& input() const { return _input; }
    padding& input_padding() { return _input_padding; }
    const padding& input_padding() const { return _input_padding; }
    padding& output_padding() { return _output_padding; }
    const padding& output_padding() const { return _output_padding; }

    virtual const CLDNN_PRIMITIVE_DESC(primitive)* get_dto() const = 0;

    std::vector<primitive_id> dependecies() const
    {
        auto result = input().store();
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
    const primitive_type_id _type;
    const primitive_id _id;

    details::primitive_id_arr _input;
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
        update_dto(*_dto.get());
        return reinterpret_cast<const CLDNN_PRIMITIVE_DESC(primitive)*>(_dto.get());
    }

protected:
    explicit primitive_base(
        const primitive_id& id,
        const std::vector<primitive_id>& input,
        const padding& input_padding = padding(),
        const padding& output_padding = padding())
        : primitive(PType::type_id(), id, input, input_padding, output_padding)
        , _dto{ new DTO }
    {}

    primitive_base(const DTO* dto)
        : primitive(reinterpret_cast<const CLDNN_PRIMITIVE_DESC(primitive)*>(dto))
        , _dto{ new DTO }
    {
        if (dto->type != PType::type_id()) 
            throw std::invalid_argument("DTO type mismatch");
    }

    virtual void update_dto(DTO& dto) const
    {
        dto.id = _id.c_str();
        dto.type = _type;
        dto.input = _input.ref();
        dto.input_padding = _input_padding;
        dto.output_padding = _output_padding;
    }
   
private:
    std::unique_ptr<DTO> _dto;
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
