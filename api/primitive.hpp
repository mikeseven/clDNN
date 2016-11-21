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
#include <memory>

namespace cldnn
{
enum class padding_types { zero, one, two };

enum primitive_types
{
    reorder,
    mean_substract,
    convolution,
    fully_connected,
    activation,
    pooling,
    normalization,
    softmax,
    depth_concat,
    data,
    input
};

typedef std::string primitive_id;
typedef string_ref primitive_id_ref;

#define BEGIN_DTO(p_type) struct p_type##_dto {\
    primitive_types type;\
    array_ref<primitive_id_ref> input;\
    tensor input_offset;\
    tensor output_offset;\
    padding_types padding_type;\

#define END_DTO(p_type) };\
static_assert(std::is_standard_layout<p_type##_dto>::value, "class has to be 'standart layout'");

BEGIN_DTO(primitive)
END_DTO(primitive)

template<primitive_types Ptype> struct primitive_type_traits;

template<typename RefElemTy, typename StorElemTy,
    typename Dummy=std::enable_if<std::is_convertible<RefElemTy, StorElemTy>::value && std::is_convertible<StorElemTy, RefElemTy>::value> >
class array_ref_store
{
public:
    array_ref_store(const std::vector<StorElemTy>& arr)
        :_data_store(arr)
    {
        for (auto& s : _data_store)
        {
            _arr_store.push_back(s);
        }
    }

    array_ref_store(array_ref<RefElemTy> arr)
    {
        for(auto& ref:arr)
        {
            _data_store.push_back(ref);
        }
        for (auto& s : _data_store)
        {
            _arr_store.push_back(s);
        }
    }

    operator array_ref<RefElemTy>() const { return _arr_store; }
private:
    std::vector<StorElemTy> _data_store;
    std::vector<RefElemTy> _arr_store;
};

typedef array_ref_store<primitive_id_ref, primitive_id> primitive_id_arr;

struct primitive_desc
{
    virtual const primitive_dto* get_dto() const = 0;
    virtual ~primitive_desc() = default;
};

template<primitive_types PType>
class primitive_desc_base : public primitive_desc
{
public:
    typedef typename primitive_type_traits<PType>::dto_type dto_type;
    typedef typename primitive_type_traits<PType>::desc_type desc_type;

    primitive_desc_base(const primitive_dto* dto)
        :_inputs(dto->input)
    {
        if (dto->type != PType) throw std::runtime_error("primitive type does not match");
        std::memcpy(dto, &_dto, sizeof(_dto));
        _dto.input = _inputs;
    }

    const primitive_dto* get_dto() const override { return reinterpret_cast<primitive_dto*>(&_dto); }

protected:
    explicit primitive_desc_base(const std::vector<primitive_id>& inputs)
        :_inputs(inputs)
    {
        _dto.type = PType;
        _dto.input = _inputs;
    }

    primitive_id_arr _inputs;
    typename primitive_type_traits<PType>::dto_type _dto;
};

#define BEGIN_DESC(p_type) class p_type##_desc;\
template<> struct primitive_type_traits<p_type>\
{\
    typedef p_type##_dto dto_type;\
    typedef p_type##_desc desc_type;\
};\
class p_type##_desc : public primitive_desc_base<p_type> {

#define END_DESC(p_type) };


}
