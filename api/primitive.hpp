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
#include <cstdint>
#include "cldnn_defs.h"
#include "compounds.h"
#include "tensor.hpp"
#include <string>
#include <memory>
#include <iterator>
#include <cmath>

namespace cldnn
{
struct padding
{
    enum types : uint32_t
    {
        zero, one, two
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

API_CLASS(padding)

typedef const struct primitive_type* primitive_type_id;
typedef std::string primitive_id;
typedef string_ref primitive_id_ref;

/**
 * \brief Template class to store data to be referenced by array_ref<RefElemTy>
 * \tparam RefElemTy Reference type. e.g. string_ref
 * \tparam StorElemTy Storage type. e.g. std::string
 * \tparam Dummy RefElemTy and StorElemTy should be implicitly convertible
 */
template<typename RefElemTy, typename StorElemTy,
    typename Dummy=std::enable_if<std::is_convertible<RefElemTy, StorElemTy>::value && std::is_convertible<StorElemTy, RefElemTy>::value> >
class array_ref_store
{
public:
    typedef StorElemTy value_type;
    array_ref_store(const std::vector<StorElemTy>& arr)
        :_data_store(arr)
    {
        update_ref();
    }

    array_ref_store(array_ref<RefElemTy> arr)
    {
        //Fill _data_store by copies of strings referenced in arr
        std::copy(arr.begin(), arr.end(), std::back_inserter(_data_store));
        update_ref();
    }

    void push_back(const StorElemTy& Val)
    {
        _data_store.push_back(Val);
        update_ref();
    }

    size_t size() const
    {
        assert(_data_store.size() == _ref_store.size());
        return _data_store.size();
    }

    // explicit conversion
    const std::vector<RefElemTy>& ref() const { return _ref_store; }
    const std::vector<StorElemTy>& store() const { return _data_store; }

    // implicit conversion
    operator array_ref<RefElemTy>() const { return ref(); }
    operator const std::vector<StorElemTy>&() const { return _data_store; }
private:
    std::vector<StorElemTy> _data_store;
    std::vector<RefElemTy> _ref_store;

    void update_ref()
    {
        //fill _ref_store by references to strings in _data_store
        std::copy(_data_store.begin(), _data_store.end(), std::back_inserter(_ref_store));
    }
};

#define BEGIN_DTO(PType) struct PType##_dto {\
    primitive_type_id type;\
    primitive_id_ref id;\
    array_ref<primitive_id_ref> input;\
    padding input_padding;\
    padding output_padding;\
    primitive_dto* as_base() { return reinterpret_cast<primitive_dto*>(this); }\
    const primitive_dto* as_base() const { return reinterpret_cast<const primitive_dto*>(this); }\

#define END_DTO(PType) };\
static_assert(std::is_standard_layout<PType##_dto>::value, "class has to be 'standart layout'");

#define DTO(PType) PType##_dto

BEGIN_DTO(primitive)
template<class PType>
typename PType::dto* as()
{
    if (type != PType::type_id()) throw std::invalid_argument("type");
    return reinterpret_cast<typename PType::dto*>(this);
}
template<class PType>
const typename PType::dto* as() const
{
    if (type != PType::type_id()) throw std::invalid_argument("type");
    return reinterpret_cast<const typename PType::dto*>(this);
}
END_DTO(primitive)

struct primitive
{
    virtual const primitive_dto* get_dto() const = 0;
    virtual primitive_type_id type() const = 0;
    virtual primitive_id id() const = 0;
    virtual std::vector<primitive_id> dependecies() const = 0;
    virtual padding input_padding() const = 0;
    virtual padding output_padding() const = 0;
    virtual ~primitive() = default;
    operator primitive_id() const { return id(); }

    //TODO remove backward compatibility
    tensor input_offset() const { return input_padding().size().negate(); }
    tensor output_offset() const { return output_padding().size(); }
    padding::types padding_type() const { return input_padding().type(); }
};

typedef array_ref_store<primitive_id_ref, primitive_id> primitive_id_arr;

template<class PType, class DTO>
class primitive_base : public primitive
{
public:
    const primitive_dto* get_dto() const override { return reinterpret_cast<const primitive_dto*>(&_dto); }

    primitive_id id() const override { return _id; }
    const std::vector<primitive_id>& input() const
    {
        return _input;
    }

    std::vector<primitive_id> dependecies() const override
    {
        auto result = input();
        auto deps = get_dependencies();
        result.insert(result.end(), deps.begin(), deps.end());
        return result;
    }

    primitive_type_id type() const override { return _dto.type; }
    padding input_padding() const override { return _dto.input_padding; }
    padding output_padding() const override { return _dto.output_padding; }

protected:
    template<typename ...Args>
    explicit primitive_base(
        const primitive_id& id,
        const std::vector<primitive_id>& input,
        const padding& input_padding =  padding(),
        const padding& output_padding = padding(),
        Args... args)
        : _id(id), _input(input)
        , _dto{ PType::type_id(), _id, _input, input_padding, output_padding, args... }
    {}

    primitive_base(const DTO* dto)
        : _id(dto->id), _input(dto->input)
        , _dto{ *dto }
    {
        if (_dto.type != PType::type_id()) throw std::invalid_argument("DTO type mismatch");
        _dto.id = _id;
        _dto.input = _input;
    }

    virtual std::vector<primitive_id> get_dependencies() const { return{}; }

    const primitive_id _id;
    const primitive_id_arr _input;
    DTO _dto;
};

}
