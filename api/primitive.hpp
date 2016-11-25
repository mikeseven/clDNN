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
#include "memory.hpp"
#include <string>
#include <memory>
#include <iterator>

namespace cldnn
{
enum class padding_types { zero, one, two };

//enum primitive_types
//{
//    reorder,
//    mean_substract,
//    convolution,
//    fully_connected,
//    activation,
//    pooling,
//    normalization,
//    softmax,
//    depth_concat,
//    data,
//    input
//};

//template<primitive_types Ptype> struct primitive_type_traits{};

typedef std::string primitive_id;
typedef const struct primitive_type_impl* primitive_type;
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
    array_ref_store(const std::vector<StorElemTy>& arr)
        :_data_store(arr) //Not sure if it will work correctly: , _arr_store(_data_store)
    {
        //fill _arr_store by references to strings in _data_store
        std::copy(_data_store.begin(), _data_store.end(), std::back_inserter(_arr_store));
    }

    array_ref_store(array_ref<RefElemTy> arr)
    {
        //Fill _data_store by copies of strings referenced in arr
        std::copy(arr.begin(), arr.end(), std::back_inserter(_data_store));
        //fill _arr_store by references to strings in _data_store
        std::copy(_data_store.begin(), _data_store.end(), std::back_inserter(_arr_store));
    }

    void push_back(const StorElemTy& Val)
    {
        _data_store.push_back(Val);
        _arr_store.push_back(_data_store.back());
    }

    size_t size() const
    {
        assert(_data_store.size() == _arr_store.size());
        return _data_store.size();
    }

    // explicit conversion
    const std::vector<RefElemTy>& ref() const { return _arr_store; }
    const std::vector<StorElemTy>& store() const { return _data_store; }

    // implicit conversion
    operator array_ref<RefElemTy>() const { return _arr_store; }
    operator const std::vector<StorElemTy>&() const { return _data_store; }
private:
    std::vector<StorElemTy> _data_store;
    std::vector<RefElemTy> _arr_store;
};

#define BEGIN_DTO(PType) struct PType##_dto {\
    primitive_type type;\
    primitive_id_ref id;\
    array_ref<primitive_id_ref> input;\
    tensor input_offset;\
    tensor output_offset;\
    padding_types padding_type;\
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
    virtual primitive_type type() const = 0;
    virtual primitive_id id() const = 0;
    virtual std::vector<primitive_id> input() const = 0;
    virtual tensor input_offset() const = 0;
    virtual tensor output_offset() const = 0;
    virtual padding_types padding_type() const = 0;
    virtual ~primitive() = default;
};

typedef array_ref_store<primitive_id_ref, primitive_id> primitive_id_arr;

template<class PType, class DTO>
class primitive_base : public primitive
{
public:
    const primitive_dto* get_dto() const override { return reinterpret_cast<primitive_dto*>(&_dto); }

    primitive_id id() const override { return _id; }
    std::vector<primitive_id> input() const override { return _input; }
    primitive_type type() const override { return _dto.type; }
    tensor input_offset() const override { return _dto.input_offset; }
    tensor output_offset() const override { return _dto.output_offset; }
    padding_types padding_type() const override { return _dto.padding_type; }

protected:
    template<typename ...Args>
    explicit primitive_base(
        const primitive_id& id,
        const std::vector<primitive_id>& input,
        const tensor& input_offset = { format::x,{ 0 } },
        const tensor& output_offset = { format::x,{ 0 } },
        const padding_types padding_type = padding_types::zero,
        Args... args)
        : _id(id), _input(input)
        , _dto{ PType::type_id(), _id, _input, input_offset, output_offset, padding_type, args }
    {}

    primitive_base(const DTO* dto)
        : _id(dto->id), _input(dto->input)
        , _dto{ *dto }
    {
        if (dto->type != PType::type_id()) throw std::invalid_argument("DTO type mismatch");
        dto->id = _id;
        dto->input = _input;
    }

    primitive_id _id;
    primitive_id_arr _input;
    DTO _dto;
};

}
