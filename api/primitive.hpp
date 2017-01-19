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
#include <string>

namespace cldnn
{
struct padding
{
    enum types : int32_t
    {
        zero = cldnn_padding_zero,
        one = cldnn_padding_one,
        two = cldnn_padding_two,
    };

    types type() const { return _type; };
    tensor size() const { return _size; };
    padding(format format, const std::vector<tensor::value_type>& sizes, types type = zero )
        : _size(format, 0, to_abs(sizes)), _type(type)
    {}

    padding(): padding(format::x, {0}, zero){}

    padding(const padding& other)
        : _size(other._size), _type(other._type)
    {}

    padding(const cldnn_padding& other)
        : _size(other.size), _type(static_cast<types>(other.type))
    {}

    operator cldnn_padding() const
    {
        return{ static_cast<cldnn_tensor>(_size), static_cast<cldnn_padding_type>(_type) };
    }

    padding& operator=(const padding& other)
    {
        if (this == &other)
            return *this;
        _size = other._size;
        _type = other._type;
        return *this;
    }

private:
    tensor _size;
    types _type;

    static std::vector<tensor::value_type> to_abs(const std::vector<tensor::value_type>& sizes)
    {
        std::vector<tensor::value_type> result(sizes.size());
        for(size_t i = 0; i < result.size(); i++)
        {
            result[i] = abs(sizes[i]);
        }
        return std::move(result);
    }
};

CLDNN_API_CLASS(padding)

// TODO check this:
using primitive_type_id = cldnn_primitive_type_id;
using primitive_id_ref = cldnn_primitive_id;
using primitive_id = std::string;

namespace detail
{
class primitive_id_arr
{
public:
    typedef primitive_id value_type;
    primitive_id_arr(const std::vector<primitive_id>& arr)
        :_data_store(arr), _ref_store(create_ref(_data_store))
    {}

    primitive_id_arr(const cldnn_primitive_id_arr& arr)
        : primitive_id_arr(create_store(arr))
    {}

    primitive_id_arr(const primitive_id_arr& other)
        : _data_store(other._data_store), _ref_store(create_ref(_data_store))
    {}

    size_t size() const
    {
        return _data_store.size();
    }

    // explicit conversion
    cldnn_primitive_id_arr ref() const
    {
        return{ _ref_store.data(), _ref_store.size() };
    }
    const std::vector<primitive_id>& store() const { return _data_store; }

    // implicit conversion
    operator const std::vector<primitive_id>&() const { return _data_store; }
private:
    const std::vector<primitive_id> _data_store;
    const std::vector<cldnn_primitive_id> _ref_store;

    static std::vector<cldnn_primitive_id> create_ref(const std::vector<primitive_id>& store)
    {
        //fill _ref_store by references to strings in _data_store
        std::vector<cldnn_primitive_id> result(store.size());
        for (size_t i = 0; i < store.size(); i++)
        {
            result[i] = store[i].c_str();
        }
        return std::move(result);
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
    const std::vector<primitive_id>& input() const { return _input; }
    const padding& input_padding() const { return _input_padding; }
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
    tensor input_offset() const { return input_padding().size().negate(); }
    tensor output_offset() const { return output_padding().size(); }
    padding::types padding_type() const { return input_padding().type(); }
protected:
    const primitive_type_id _type;
    const primitive_id _id;
    const detail::primitive_id_arr _input;
    const padding _input_padding;
    const padding _output_padding;

    virtual std::vector<primitive_id> get_dependencies() const { return{}; }
};

template<class PType, class DTO>
class primitive_base : public primitive
{
public:
    const CLDNN_PRIMITIVE_DESC(primitive)* get_dto() const override { return reinterpret_cast<const CLDNN_PRIMITIVE_DESC(primitive)*>(&_dto); }


protected:
    template<typename ...Args>
    explicit primitive_base(
        const primitive_id& id,
        const std::vector<primitive_id>& input,
        const padding& input_padding =  padding(),
        const padding& output_padding = padding(),
        Args... args)
        : primitive(PType::type_id(), id, input, input_padding, output_padding)
        , _dto{ _type, _id.c_str(), _input.ref(), _input_padding, _output_padding, args... }
    {}

    primitive_base(const DTO* dto)
        : primitive(reinterpret_cast<const CLDNN_PRIMITIVE_DESC(primitive)*>(dto))
        , _dto{ *dto }
    {
        if (_dto.type != PType::type_id()) throw std::invalid_argument("DTO type mismatch");
        _dto.id = _id.c_str();
        _dto.input = _input.ref();
    }

    DTO _dto;
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
